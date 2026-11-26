import AppKit
import Foundation
import Vision

let arguments = CommandLine.arguments
let jsonOutput = arguments.count == 3 && arguments[1] == "--json"
guard arguments.count == 2 || jsonOutput else {
    fputs("usage: swift ocr_crossover_round70.swift [--json] page.png\n", stderr)
    exit(2)
}
let path = arguments[arguments.count - 1]
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
struct RecognizedLine: Codable {
    let text: String
    let x: Double
    let y: Double
    let width: Double
    let height: Double
}
var lines: [RecognizedLine] = []
for observation in request.results ?? [] {
    if let candidate = observation.topCandidates(1).first {
        let box = observation.boundingBox
        lines.append(RecognizedLine(text: candidate.string, x: box.origin.x, y: box.origin.y,
                                    width: box.size.width, height: box.size.height))
    }
}
if jsonOutput {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let data = try encoder.encode(lines)
    print(String(data: data, encoding: .utf8)!)
} else {
    for line in lines {
        print(line.text)
    }
}
