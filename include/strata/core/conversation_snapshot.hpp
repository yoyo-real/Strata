#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/conversation_file.hpp"
#include "strata/core/layer.hpp"
#include "strata/core/session.hpp"

#include <string>

namespace strata::core {

// Caller synchronizes the device before saving, and after restoring all layers.
// include_index is false for the draft layer (its attention has no indexer).
size_t conversation_kv_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index);
// A nonzero unchanged_tokens is valid only for storage retained from an image
// actually restored into this session, bounded by every subsequent rewrite.
// Equal token IDs alone do not establish that its K/V bytes are unchanged.
bool conversation_kv_save(ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                          int64_t upto, bool include_index, std::string& error,
                          int64_t unchanged_tokens = 0, size_t* reused_bytes = nullptr);
bool conversation_kv_capture_bytes(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                                   int64_t upto, bool include_index, size_t& bytes, std::string& error);
// No CUDA calls or destination writes. Used for whole-session prevalidation.
bool conversation_kv_validate(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                              int64_t upto, bool include_index, std::string& error);
bool conversation_kv_restore(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                             int64_t upto, bool include_index, std::string& error);
// Diagnostic read-back after a synchronized restore. Uses 64 KiB of stack
// workspace, compares authoritative bytes and resident draft-ring pages, and
// fingerprints the authoritative payload only. Never changes model state.
bool conversation_kv_verify(const ConversationKv& image, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, uint64_t& fingerprint, std::string& error);
// Disk save without a host copy: the layer's authoritative K/V as a streamed source (same bytes and metadata as
// conversation_kv_save).  Caller synchronizes the device first and keeps the state untouched while it is read.
bool conversation_kv_source(SessionKvSource& source, const QsaState& state, const ModelGeometry& g,
                            int64_t upto, bool include_index, std::string& error);
// A K/V layer restored from its file without a host copy (disk parking, ConversationKvPending): the header is
// checked as conversation_kv_validate checks an image's, each part's size before its bytes, and the bytes go to the
// authoritative pools as they arrive (bounds-checked; synchronous, so the caller's buffer is free on return).  _done
// does what conversation_kv_restore does after its copies (the VRAM residency); with upto = 0 it only drops it.
bool conversation_kv_stream_layer(const ConversationKv& header, const QsaState& state, const ModelGeometry& g,
                                  int64_t upto, bool include_index, std::string& error);
bool conversation_kv_stream_part(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                 size_t part, uint64_t bytes, std::string& error);
bool conversation_kv_stream_bytes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                  size_t part, uint64_t at, const uint8_t* data, size_t n, std::string& error);
bool conversation_kv_stream_done(const QsaState& state, const ModelGeometry& g, int64_t upto, std::string& error);
// The bytes of each K/V part (k, v, k_scale, v_scale, pooled) a snapshot of `upto` tokens holds: no state read.
bool conversation_kv_part_sizes(const QsaState& state, const ModelGeometry& g, int64_t upto, bool include_index,
                                std::array<uint64_t, 5>& sizes, std::string& error);
// Disk sessions: the read limits this session can ever restore - its geometry and layer range, at most
// min(max_tokens, its cells) tokens, `max_checkpoints` checkpoints, the exact running-state sizes and the K/V part
// sizes at that many tokens - so session_file_read refuses an oversized or foreign file before it allocates.
bool conversation_session_read_limits(SessionReadLimits& limits, const SessionState& session, const ModelGeometry& g,
                                      const QsaState& draft, uint64_t max_tokens, uint64_t max_checkpoints,
                                      std::string& error);

struct ConversationStateSizes {
    size_t gdn = 0, ple = 0, tail = 0, dead = 0, block_pos = 0;
};
/// Whole-model sizes: `gdn` covers every GDN layer, the indexer sizes are per QSA layer.
bool conversation_state_sizes(const ModelGeometry& g, ConversationStateSizes& sizes, std::string& error);
/// The same for one session's layer carve (#216): `gdn` covers its `gdn_alloc` rows; the per-QSA-layer sizes
/// apply to each owned state [qsa_ord0, qsa_ord0 + qsa_alloc).  Rejects an inconsistent carve.  Checkpoints,
/// snapshots and their validation all use this: a session saves and restores only the state it owns.
bool conversation_session_sizes(const ModelGeometry& g, const SessionState& session, ConversationStateSizes& sizes,
                                std::string& error);
bool conversation_checkpoint_validate(const ConversationCheckpoint& checkpoint, const SessionState& session,
                                      const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_save(ConversationCheckpoint& checkpoint, const SessionState& session,
                                  const ModelGeometry& g, std::string& error);
bool conversation_checkpoint_restore(const ConversationCheckpoint& checkpoint, SessionState& session,
                                     const ModelGeometry& g, std::string& error);

struct ConversationView {
    const std::vector<int32_t>& ids;
    const std::vector<ConversationImageKey>& images;
    const std::vector<ConversationCheckpoint>& checkpoints;
    bool cvec;
};
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session,
                                 const ModelGeometry& g, const QsaState& draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g,
                                         const QsaState& draft, size_t& bytes, std::string& error);
// The capture estimate includes retained capacity and transient segment directories;
// only estimate - reuse.bytes() requires additional physical RAM. Capture consumes
// the uniquely owned reusable buffers, including on failure.
// Caller admits the estimate before invoking capture. Allocation failures propagate
// to the RAM policy; the active session is never modified by capture.
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view,
                                const SessionState& session, const ModelGeometry& g,
                                const QsaState& draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
// Disk save without capturing the K/V on the host: `meta` gets everything but the K/V (running state copied,
// checkpoints as given by the view), `sources` one streamed source per QSA layer then the draft.  Caller has
// synchronized and must not run the session until the file is written.
bool conversation_snapshot_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                   const ConversationView& view, const SessionState& session,
                                   const ModelGeometry& g, const QsaState& draft, std::string& error);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState& draft, std::string& error);
// Disk parking's capture: the same as conversation_snapshot_sources, with the draft layer optional (null: no draft
// K/V, as conversation_snapshot_save takes it) and the checkpoints MOVED from `checkpoints` (which must be the view's)
// into `meta` instead of copied - the session's own, lent to the file write; the caller moves them back after it.
// On failure nothing is moved.
bool conversation_snapshot_park_sources(SavedConversation& meta, std::vector<SessionKvSource>& sources,
                                        const ConversationView& view, std::vector<ConversationCheckpoint>& checkpoints,
                                        const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                        std::string& error);
// file_failed: an image whose K/V was still in its file (kv_pending) turned out bad or unreadable while it was
// streamed into the K/V pools.  The pools are partly written and the running state is not restored: the session
// holds no conversation, and the caller reads the prompt from token 0 (the outgoing one was parked first).
enum class ConversationRestore { restored, invalid, transfer_failed, file_failed };
// Invalid images are rejected before any CUDA call/write. Transfer failure may
// leave partial state: caller MUST NOT continue inference from that session.
// An image with kv_pending is validated without its K/V (its layer count only); each layer is checked as it is read.
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState& draft,
                                                   std::string& error);

// The same with `draft == nullptr`: an image WITHOUT the draft layer's K/V (kv holds the session's own QSA layers
// only).  A layer split's later stages park this way; the draft ring is saved once, with the first stage's image.
// An image is validated and restored with the same kind of call it was saved with (a K/V layer count mismatch is
// rejected as invalid).
bool conversation_snapshot_bytes(const ConversationView& view, const SessionState& session, const ModelGeometry& g,
                                 const QsaState* draft, size_t& bytes, std::string& error);
bool conversation_snapshot_capture_bytes(const ConversationKvReuse& reuse, const ConversationView& view,
                                         const SessionState& session, const ModelGeometry& g, const QsaState* draft,
                                         size_t& bytes, std::string& error);
bool conversation_snapshot_save(SavedConversation& image, const ConversationView& view, const SessionState& session,
                                const ModelGeometry& g, const QsaState* draft, std::string& error,
                                ConversationKvReuse reuse = {}, size_t* reused_bytes = nullptr);
bool conversation_snapshot_validate(const SavedConversation& image, const SessionState& session,
                                    const ModelGeometry& g, const QsaState* draft, std::string& error);
ConversationRestore conversation_snapshot_restore(const SavedConversation& image, SessionState& session,
                                                   const ModelGeometry& g, const QsaState* draft, std::string& error);

} // namespace strata::core
