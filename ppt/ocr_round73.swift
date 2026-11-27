import AppKit
import Foundation
import Vision

guard CommandLine.arguments.count == 2 else {
    fputs("usage: swift ocr_round73.swift page.png\n", stderr)
    exit(2)
}
let path = CommandLine.arguments[1]
guard let image = NSImage(contentsOfFile: path) else {
    fputs("cannot read image: \(path)\n", stderr)
    exit(2)
}
var rectangle = CGRect(origin: .zero, size: image.size)
guard let cgImage = image.cgImage(forProposedRect: &rectangle, context: nil, hints: nil) else {
    fputs("cannot decode image: \(path)\n", stderr)
    exit(2)
}
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.recognitionLanguages = ["zh-Hans", "en-US"]
request.usesLanguageCorrection = false
request.minimumTextHeight = 0.008
let handler = VNImageRequestHandler(cgImage: cgImage, options: [:])
try handler.perform([request])
for case let observation as VNRecognizedTextObservation in request.results ?? [] {
    if let candidate = observation.topCandidates(1).first {
        print(candidate.string)
    }
}
