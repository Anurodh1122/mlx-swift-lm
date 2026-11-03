// PipelineSlice.swift — layer-range slicing for pipeline-split inference (llm-mesh)
//
// A "slice" is a contiguous range of transformer layers [start, end) that this process
// owns. Stage 0 (start == 0) embeds token ids; later stages accept a hidden-state tensor
// [batch, seq, hidden] instead. The last stage (end == total) applies the final norm and
// lm_head; earlier stages return the raw hidden state to be shipped to the next stage.
// With no slice configured every model behaves exactly as before.

import Foundation
import MLX
import MLXNN

public final class PipelineSliceConfig: @unchecked Sendable {
    public static let shared = PipelineSliceConfig()
    /// Half-open layer range owned by this process, or nil for the full model.
    public var range: Range<Int>?
    private init() {}

    public var isActive: Bool { range != nil }

    /// Parse "a:b" (half-open) into a range.
    public static func parse(_ spec: String) -> Range<Int>? {
        let p = spec.split(separator: ":")
        guard p.count == 2, let a = Int(p[0]), let b = Int(p[1]), a >= 0, b > a else { return nil }
        return a ..< b
    }

    /// Effective range for a model with `total` layers (clamped), nil if no slice.
    public func effectiveRange(total: Int) -> Range<Int>? {
        guard let r = range else { return nil }
        let lo = min(max(0, r.lowerBound), total)
        let hi = min(max(lo, r.upperBound), total)
        return lo ..< hi
    }

    /// Layer index encoded in a tensor key (".layers.N."), if any.
    static func layerIndex(of key: String) -> Int? {
        guard let m = key.range(of: #"\.layers\.(\d+)\."#, options: .regularExpression) else { return nil }
        return Int(key[m].dropFirst(8).dropLast())
    }

    /// Read-time filter: true if this tensor belongs to a layer outside the slice.
    /// Needs no total layer count, so it can run before any tensor is loaded.
    public func dropsLayer(key: String) -> Bool {
        guard let r = range, let idx = Self.layerIndex(of: key) else { return false }
        return !r.contains(idx)
    }

    /// True if a checkpoint tensor key belongs to a layer outside the slice (or to the
    /// lm_head when this is not the last stage) and should not be loaded.
    public func shouldDrop(key: String, totalLayers: Int) -> Bool {
        guard let r = effectiveRange(total: totalLayers) else { return false }
        if let m = key.range(of: #"\.layers\.(\d+)\."#, options: .regularExpression) {
            let digits = key[m].dropFirst(8).dropLast()  // strip ".layers." and trailing "."
            if let idx = Int(digits) { return !r.contains(idx) }
        }
        if key.contains("lm_head.") && r.upperBound < totalLayers { return true }
        return false
    }
}

/// Models that can run only a slice of their layers.
public protocol PipelineSliceable: AnyObject {
    var pipelineRange: Range<Int>? { get set }
}

/// After weights are applied, replace every parameter this slice does not use with a scalar.
/// Un-loaded layers still hold lazily-initialised full-size placeholders, and `eval(model)`
/// would otherwise materialise them (full fp16 layers, plus lm_head on non-final stages).
public func releaseOutOfSliceParameters(_ model: Module) {
    let cfg = PipelineSliceConfig.shared
    guard let r = cfg.range else { return }
    let flat = model.parameters().flattened()
    let total = (flat.compactMap { PipelineSliceConfig.layerIndex(of: $0.0) }.max() ?? -1) + 1
    var replacement = [String: MLXArray]()
    for (key, _) in flat {
        let outOfLayers = cfg.dropsLayer(key: key)
        let unusedHead = key.contains("lm_head.") && r.upperBound < total
        let unusedEmbed = key.contains("embed_tokens.") && r.lowerBound > 0 && r.upperBound < total
        if outOfLayers || unusedHead || unusedEmbed { replacement[key] = MLXArray(0) }
    }
    if !replacement.isEmpty {
        try? model.update(parameters: ModuleParameters.unflattened(replacement), verify: [])
    }
}
