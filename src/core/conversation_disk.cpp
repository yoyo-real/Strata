// The disk half of --conversation-cache-storage disk (conversation_disk.hpp): entry bookkeeping, LRU eviction and
// the budget, over files the installed ConversationDiskIo writes and reads.  Pure host code: no CUDA.
#include "strata/core/conversation_disk.hpp"

#include <cstdio>

#ifdef _WIN32
#include <process.h>
#else
#include <sys/types.h>
#include <unistd.h>
#endif

namespace strata::core {

namespace {
uint64_t this_pid() {
#ifdef _WIN32
    return (uint64_t) _getpid();
#else
    return (uint64_t) ::getpid();
#endif
}
} // namespace

std::string ConversationDiskCache::next_path() {
    char leaf[48];
    std::snprintf(leaf, sizeof leaf, "park-%llu-%06llu", (unsigned long long) this_pid(), (unsigned long long) seq_++);
    return dir_ + "/" + leaf;
}

bool ConversationDiskCache::make_room(size_t incoming, size_t held) {
    (void) held;   // take() already removed the restored entry's file: no disk budget is held against this park
    if (!enabled() || incoming > budget_) return false;
    while (!entries_.empty() && (entries_.size() >= slots_ || bytes_ > budget_ - incoming)) {
        // the oldest entry that does not hold a pinned shared prefix leaves; with only pinned ones left the new
        // image does not fit, exactly as the RAM cache's loop
        if (!evict_oldest()) return false;
    }
    return true;
}

bool ConversationDiskCache::evict_oldest() {
    auto victim = std::find_if(entries_.begin(), entries_.end(), [](const Entry& e) { return !e.pinned(); });
    if (victim == entries_.end()) return false;
    remove_file(*victim);
    bytes_ -= victim->bytes;
    entries_.erase(victim);
    ++evictions_;
    return true;
}

size_t ConversationDiskCache::clear() {
    const size_t n = entries_.size();
    for (const auto& e : entries_) remove_file(e);
    entries_.clear();
    bytes_ = 0;
    return n;
}

size_t ConversationDiskCache::drop_superseded(const std::vector<int32_t>& ids,
                                              const std::vector<ConversationImageKey>& images,
                                              const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
    auto held = [&](const ConversationCheckpoint& c) {
        if (c.ids == ids && c.imgs == images) return true;
        for (const auto& k : checkpoints)
            if (k.ids == c.ids && k.imgs == c.imgs) return true;
        return false;
    };
    size_t dropped = 0;
    for (size_t i = 0; i < entries_.size();) {
        const Entry& e = entries_[i];
        const ConversationCheckpoint* deepest = nullptr;
        for (const auto& c : e.checkpoints)
            if (!deepest || c.ids.size() > deepest->ids.size()) deepest = &c;
        if (e.cvec == cvec && deepest && !deepest->ids.empty() && held(*deepest)) {
            remove_file(e);
            bytes_ -= e.bytes;
            entries_.erase(entries_.begin() + (std::ptrdiff_t) i);
            ++dropped;
            continue;
        }
        ++i;
    }
    superseded_ += dropped;
    return dropped;
}

std::optional<SavedConversation> ConversationDiskCache::take(size_t index, bool keep) {
    if (index >= entries_.size()) return std::nullopt;
    Entry e = std::move(entries_[index]);
    entries_.erase(entries_.begin() + (std::ptrdiff_t) index);
    bytes_ -= e.bytes;
    SavedConversation image;
    size_t file_bytes = 0;
    last_error_.clear();
    if (!io_.read) {
        last_error_ = "no read hook installed";
        remove_file(e);
        return std::nullopt;
    }
    if (!io_.read(e.path, image, file_bytes, last_error_)) {
        remove_file(e);   // unreadable: gone for good, the caller reads the prompt again
        return std::nullopt;
    }
    if (keep) {   // the parked conversation stays whole: back in the index as the most recently used
        e.reading = image.kv_pending;
        bytes_ += e.bytes;
        entries_.push_back(e);
    } else if (image.kv_pending) {
        // restored: the conversation is live again, its parked file goes - once its K/V is read
        image.kv_pending->remove_file_when_done();
    } else {
        unlink_file(e.path);
    }
    // the pin marks are run-time only - the file does not carry them (conversation_file.hpp), so the index's do
    for (const auto& m : e.checkpoints)
        for (auto& c : image.checkpoints)
            if (c.ids == m.ids && c.imgs == m.imgs) c.pinned = m.pinned;
    return image;
}

bool ConversationDiskCache::forget_file(const std::string& path) {
    for (auto it = entries_.begin(); it != entries_.end(); ++it) {
        if (it->path != path) continue;
        remove_file(*it);
        bytes_ -= it->bytes;
        entries_.erase(it);
        return true;
    }
    return false;
}

bool ConversationDiskCache::put(SavedConversation&& image, size_t held) {
    (void) held;   // as in make_room: nothing of this exchange is on the disk's books
    const size_t n = image.bytes();
    if (!enabled() || n > budget_) return false;
    drop_superseded(image.live.ids, image.live.imgs, image.checkpoints, image.cvec);
    if (!make_room(n)) return false;
    // the index metadata first, from the image still in hand
    Entry e;
    e.live = meta_only(image.live);
    for (const auto& c : image.checkpoints) e.checkpoints.push_back(meta_only(c));
    e.cvec = image.cvec;
    e.path = next_path();
    size_t file_bytes = 0;
    last_error_.clear();
    if (!io_.write) {
        last_error_ = "no write hook installed";
        return false;
    }
    if (!io_.write(e.path, image, file_bytes, last_error_)) return false;   // the caller logs why
    e.bytes = file_bytes > 0 ? file_bytes : n;
    bytes_ += e.bytes;
    entries_.push_back(std::move(e));
    image = {};   // the snapshot's host buffers go back to the kernel: a parked conversation holds no RAM
    return true;
}

bool ConversationDiskCache::put_streamed(const SavedConversation& meta, const std::vector<SessionKvSource>& kv) {
    last_error_.clear();
    size_t n = meta.bytes();
    for (const auto& k : kv)
        for (size_t b : k.sizes) n += b;
    if (!enabled() || n > budget_ || !meta.kv.empty()) return false;
    drop_superseded(meta.live.ids, meta.live.imgs, meta.checkpoints, meta.cvec);
    if (!make_room(n)) return false;
    Entry e;
    e.live = meta_only(meta.live);
    for (const auto& c : meta.checkpoints) e.checkpoints.push_back(meta_only(c));
    e.cvec = meta.cvec;
    e.path = next_path();
    size_t file_bytes = 0;
    if (!io_.write_sources) {
        last_error_ = "no streamed write hook installed";
        return false;
    }
    if (!io_.write_sources(e.path, meta, kv, file_bytes, last_error_)) return false;   // the caller logs why
    e.bytes = file_bytes > 0 ? file_bytes : n;
    bytes_ += e.bytes;
    entries_.push_back(std::move(e));
    return true;
}

} // namespace strata::core
