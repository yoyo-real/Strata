// File-backed parking for --serve's conversations (--conversation-cache-storage disk): the parked snapshot lives as
// a session-format file (conversation_file.hpp: format v1, temp + rename, flush, bounded read) under
// --conversation-cache-dir, and host RAM keeps only the light index it takes to match one - token ids, image keys,
// the cvec flag, the file's path and size.  The budget is --conversation-cache-mib, read as a DISK budget; the LRU,
// the slot count and the matching semantics are ConversationCache's, mirrored here entry for entry.
//
// Pure host code: no CUDA.  The file I/O itself is installed by the caller (ConversationDiskIo) so the engine can
// bind every write and read to its session identity, its read limits and its free-space preflight
// (generate.cpp); tests install the plain session_file_* calls.
#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace strata::core {

// One snapshot's file side.  `write` streams `image` to `path` (a session-format file); `read` parses `path` back
// into `image`, or all but its K/V (session_file_read_streamed: image.kv_pending).  Both report their bytes through
// `bytes`; `error` says why a failure happened.
struct ConversationDiskIo {
    std::function<bool(const std::string& path, const SavedConversation& image, size_t& bytes, std::string& error)> write;
    std::function<bool(const std::string& path, SavedConversation& image, size_t& bytes, std::string& error)> read;
    // put_streamed's write: `meta` without its K/V, which `kv` reads off the device as the file is written
    std::function<bool(const std::string& path, const SavedConversation& meta, const std::vector<SessionKvSource>& kv,
                       size_t& bytes, std::string& error)> write_sources;
};

class ConversationDiskCache {
public:
    using Match = ConversationCache::Match;

    /// `budget` is the DISK budget in bytes (the files' total size), `dir` the folder the snapshots go to.
    ConversationDiskCache(size_t budget, size_t slots, int64_t min_tokens, std::string dir)
        : budget_(budget), slots_(slots), min_tokens_(min_tokens), dir_(std::move(dir)) {}
    /// The engine installs these once, after its session identity and read limits exist.
    void io(ConversationDiskIo&& hooks) { io_ = std::move(hooks); }
    const std::string& dir() const { return dir_; }
    bool enabled() const { return budget_ != 0 && slots_ != 0 && !dir_.empty(); }
    /// --conversation-cache-min-tokens, exactly as the RAM cache reads it.
    bool wants(int64_t tokens) const { return min_tokens_ <= 0 || tokens >= min_tokens_; }
    size_t bytes() const { return bytes_; }   // the files' bytes, not any RAM
    size_t size() const { return entries_.size(); }
    size_t evictions() const { return evictions_; }
    size_t superseded() const { return superseded_; }
    size_t slots() const { return slots_; }
    // the longest parked conversation, in tokens (--kv-grow keeps the K/V that long while it could be restored)
    int64_t longest_tokens() const {
        int64_t n = 0;
        for (const auto& e : entries_) n = std::max<int64_t>(n, (int64_t) e.live.ids.size());
        return n;
    }

    // The retained K/V is a RAM-only idea: a conversation restored from disk keeps no host copy of its K/V, so its
    // next park captures fresh from the device.  These stay for call-site compatibility with ConversationCache.
    void retain(std::vector<ConversationKv>&&, int64_t, std::vector<std::vector<ConversationKv>>&& = {}) {}
    ConversationKvReuse take_reuse() { return {}; }
    size_t retained_bytes() const { return 0; }

    /// The same prefix match as ConversationCache::best: the entry whose live state or checkpoint chain holds the
    /// longest prefix of `prompt`, ties preferring the most recently parked entry.
    template<class Token>
    Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images, bool cvec) const {
        Match best;
        for (size_t i = entries_.size(); i-- > 0;) {
            const Entry& e = entries_[i];
            if (e.cvec != cvec) continue;
            auto consider = [&](const ConversationCheckpoint& c, bool live) {
                const int64_t n = conversation_prefix(c, prompt, images);
                if (n > best.tokens) best = {i, n, live};
            };
            consider(e.live, true);
            for (const auto& c : e.checkpoints) consider(c, false);
        }
        return best;
    }

    // Reads the entry's file back (through the installed io).  The entry and its file leave the cache either way:
    // a file that cannot be read is gone for good, `last_error()` says why, and the caller falls back to reading the
    // prompt - the same recoverable miss a discarded invalid snapshot is in the RAM path.  An io that leaves the K/V
    // in the file (kv_pending) hands the file over with it: it is removed when the image's K/V has been read.
    //
    // `keep`: the request resumes at one of the entry's checkpoints, not at its live state, so it continues another
    // branch and the parked conversation is still whole.  A file read leaves it as it was: the entry stays parked
    // (now the most recently used), only a file that cannot be read goes.  A RAM image would have to be copied for
    // this; a file costs nothing to keep.
    std::optional<SavedConversation> take(size_t index, bool keep = false);
    // A kept entry whose file turned out bad while its K/V was streamed (ConversationRestore::file_failed): it goes,
    // so the next match does not fail the same way.  Returns whether an entry had that file.
    bool forget_file(const std::string& path);
    const std::string& last_error() const { return last_error_; }

    // The disk budget's make_room: the same shape as the RAM one.  `held` is IGNORED: in the RAM cache it counts the
    // incoming image being exchanged (its bytes still on the books while the outgoing one parks); here take() has
    // already removed the restored entry's file, so no disk budget is held against this park.
    bool make_room(size_t incoming, size_t held = 0);
    // The least recently parked entry without a pinned shared prefix goes, file first (make_room's loop body).
    bool evict_oldest();
    // #879: every parked conversation (pinned ones too) goes.  Returns how many were parked.
    size_t clear();
    // #342: drop the parked entries an outgoing conversation supersedes (the same rule as the RAM cache).  Returns
    // how many were dropped; their files go with them.
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec);
    // Parks `image`: metadata goes to the RAM index, the snapshot to a file in dir().  After this returns true the
    // caller's copy is freed (put clears it here), so a parked conversation holds no host RAM beyond its index.
    // `held` is ignored as in make_room.  false: the budget refused it, or the write failed (the caller logs).
    bool put(SavedConversation&& image, size_t held = 0);
    // The same park with the K/V never on the host: `meta` holds everything else (meta.kv empty), `kv` reads each
    // layer off the device into the file's write buffer.  `meta` is left to the caller, whose checkpoints it may
    // hold on loan (conversation_snapshot_park_sources).
    bool put_streamed(const SavedConversation& meta, const std::vector<SessionKvSource>& kv);

private:
    // The RAM index of one parked file: the matching metadata (ids, image keys, pins), not one K/V byte.
    struct Entry {
        ConversationCheckpoint live;                     // ids + imgs only
        std::vector<ConversationCheckpoint> checkpoints; // ids + imgs + pinned only
        bool cvec = true;
        std::string path;
        size_t bytes = 0;
        // a kept entry whose file is open for a restore right now: evicting it removes the file once that read closes
        // it (a file still open cannot be removed on Windows)
        std::weak_ptr<ConversationKvPending> reading;
        bool pinned() const {
            for (const auto& c : checkpoints)
                if (c.pinned) return true;
            return false;
        }
    };
    static ConversationCheckpoint meta_only(const ConversationCheckpoint& c) {
        ConversationCheckpoint m;
        m.ids = c.ids;
        m.imgs = c.imgs;
        m.used = c.used;
        m.pinned = c.pinned;
        return m;
    }
    std::string next_path();
    void unlink_file(const std::string& path) { std::remove(path.c_str()); }
    void remove_file(const Entry& e) {
        if (auto open = e.reading.lock()) open->remove_file_when_done();
        else unlink_file(e.path);
    }

    size_t budget_ = 0, slots_ = 0, bytes_ = 0, evictions_ = 0, superseded_ = 0;
    int64_t min_tokens_ = 0;
    std::string dir_;
    ConversationDiskIo io_;
    std::deque<Entry> entries_;   // least recently parked first, as in the RAM cache
    uint64_t seq_ = 0;
    std::string last_error_;
};

// One face over the two parking storages, so the serve loop reads the same whether it parked in RAM or on disk:
// every ConversationCache method the engine calls is here, forwarded to whichever store was configured.  The RAM
// behavior is bit-for-bit the previous one (storage "ram", the default); "disk" swaps the store, not the policy.
class ConversationStore {
public:
    ConversationStore(size_t budget, size_t slots, int64_t min_tokens, const std::string& storage,
                      const std::string& dir) {
        if (storage == "disk")
            disk_.emplace(budget, slots, min_tokens, dir);
        else
            ram_ = std::make_unique<ConversationCache>(budget, slots, min_tokens);
    }
    bool disk() const { return disk_.has_value(); }
    ConversationDiskCache& disk_store() { return *disk_; }
    // The RAM store, whatever the mode: session_save_reclaim (a SAVE's RAM preflight) may evict parked RAM, never
    // parked files - they hold no host RAM - so the disk mode hands it the (empty) RAM store.
    ConversationCache& ram() { return *ram_; }

    bool enabled() const { return disk_ ? disk_->enabled() : ram_->enabled(); }
    bool wants(int64_t tokens) const { return disk_ ? disk_->wants(tokens) : ram_->wants(tokens); }
    size_t bytes() const { return disk_ ? disk_->bytes() : ram_->bytes(); }
    size_t size() const { return disk_ ? disk_->size() : ram_->size(); }
    size_t evictions() const { return disk_ ? disk_->evictions() : ram_->evictions(); }
    size_t superseded() const { return disk_ ? disk_->superseded() : ram_->superseded(); }
    size_t slots() const { return disk_ ? disk_->slots() : ram_->slots(); }
    int64_t longest_tokens() const { return disk_ ? disk_->longest_tokens() : ram_->longest_tokens(); }
    void retain(std::vector<ConversationKv>&& kv, int64_t tokens, std::vector<std::vector<ConversationKv>>&& stage_kv = {}) {
        if (disk_) disk_->retain(std::move(kv), tokens, std::move(stage_kv));
        else ram_->retain(std::move(kv), tokens, std::move(stage_kv));
    }
    void limit_reuse(int64_t first_dirty) { if (!disk_) ram_->limit_reuse(first_dirty); }
    ConversationKvReuse take_reuse() { return disk_ ? disk_->take_reuse() : ram_->take_reuse(); }
    size_t retained_bytes() const { return disk_ ? disk_->retained_bytes() : ram_->retained_bytes(); }
    bool make_room(size_t incoming, size_t held = 0) {
        return disk_ ? disk_->make_room(incoming, held) : ram_->make_room(incoming, held);
    }
    bool evict_oldest() { return disk_ ? disk_->evict_oldest() : ram_->evict_oldest(); }
    size_t clear() { return disk_ ? disk_->clear() : ram_->clear(); }
    size_t drop_superseded(const std::vector<int32_t>& ids, const std::vector<ConversationImageKey>& images,
                           const std::vector<ConversationCheckpoint>& checkpoints, bool cvec) {
        return disk_ ? disk_->drop_superseded(ids, images, checkpoints, cvec)
                     : ram_->drop_superseded(ids, images, checkpoints, cvec);
    }
    bool put(SavedConversation&& image, size_t held = 0) {
        return disk_ ? disk_->put(std::move(image), held) : ram_->put(std::move(image), held);
    }
    // disk only: the RAM store has nowhere to put a K/V it does not hold
    bool put_streamed(const SavedConversation& meta, const std::vector<SessionKvSource>& kv) {
        return disk_ && disk_->put_streamed(meta, kv);
    }
    template<class Token>
    ConversationCache::Match best(const std::vector<Token>& prompt, const std::vector<ConversationImageKey>& images,
                                  bool cvec) const {
        return disk_ ? disk_->best(prompt, images, cvec) : ram_->best(prompt, images, cvec);
    }
    // The RAM take never fails; a disk take can (an unreadable file), so both return optional and the caller falls
    // back to reading the prompt when there is nothing to restore.
    // `keep_file`: disk only (ConversationDiskCache::take); the RAM store always hands its image over
    std::optional<SavedConversation> take(size_t index, bool keep_file = false) {
        if (disk_) return disk_->take(index, keep_file);
        return ram_->take(index);
    }
    const std::string& last_error() const { return disk_ ? disk_->last_error() : empty_; }

private:
    std::unique_ptr<ConversationCache> ram_ = std::make_unique<ConversationCache>(0, 0);
    std::optional<ConversationDiskCache> disk_;
    inline static const std::string empty_;
};

} // namespace strata::core
