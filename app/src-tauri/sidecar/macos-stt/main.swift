// macOS STT sidecar: reads PCM (int16, mono) from stdin and emits JSON transcripts
import Foundation
import Speech
import AVFoundation

struct Args {
    var stdinPCM: Bool = false
    var sampleRate: Double = 16000
    var language: String = "en-US"
    // Default to false so backend can enable explicitly with --on-device
    var onDevice: Bool = false
}

func parseArgs() -> Args {
    var a = Args()
    var it = CommandLine.arguments.dropFirst().makeIterator()
    while let arg = it.next() {
        switch arg {
        case "--stdin-pcm": a.stdinPCM = true
        case "--rate": a.sampleRate = Double(it.next() ?? "16000") ?? 16000
        case "--lang": a.language = it.next() ?? "en-US"
        case "--on-device": a.onDevice = true
        default: break
        }
    }
    return a
}

let args = parseArgs()

let locale = Locale(identifier: args.language)
guard let recognizer = SFSpeechRecognizer(locale: locale) else {
    fputs("{\"error\":\"unsupported_language\"}\n", stderr)
    exit(2)
}

// Request authorization (may be already granted). Timeout quickly to avoid hanging.
let authSem = DispatchSemaphore(value: 0)
SFSpeechRecognizer.requestAuthorization { status in
    if status != .authorized {
        fputs("{\"error\":\"speech_not_authorized\"}\n", stderr)
        // Continue anyway; on-device recognition may still work in some environments
    }
    authSem.signal()
}
_ = authSem.wait(timeout: .now() + 2)

let request = SFSpeechAudioBufferRecognitionRequest()
request.shouldReportPartialResults = true
if #available(macOS 12.0, *) {
    request.requiresOnDeviceRecognition = args.onDevice
}

let queue = DispatchQueue(label: "stt.reader")
let out = FileHandle.standardOutput

var finished = false

let fmt = AVAudioFormat(commonFormat: .pcmFormatInt16, sampleRate: args.sampleRate, channels: 1, interleaved: true)!

// Start recognition task
let task = recognizer.recognitionTask(with: request) { result, error in
    if let result = result {
        let text = result.bestTranscription.formattedString
        let final = result.isFinal
        let obj = ["text": text, "final": final] as [String : Any]
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: []) {
            out.write(data)
            out.write("\n".data(using: .utf8)!)
        }
        if final { finished = true }
    } else if let error = error {
        let obj = ["error": String(describing: error)]
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: []) {
            out.write(data); out.write("\n".data(using: .utf8)!)
        }
    }
}

// Read stdin and append buffers
queue.async {
    let stdinFH = FileHandle.standardInput
    while true {
        let chunk = stdinFH.readData(ofLength: 4096)
        if chunk.count == 0 { break }

        let frameCount = chunk.count / 2
        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frameCount)) else { continue }
        buf.frameLength = AVAudioFrameCount(frameCount)
        chunk.withUnsafeBytes { rawPtr in
            guard let base = rawPtr.bindMemory(to: Int16.self).baseAddress else { return }
            buf.int16ChannelData!.pointee.assign(from: base, count: frameCount)
        }
        request.append(buf)
    }
    request.endAudio()
}

// Keep runloop alive
RunLoop.current.run(until: Date().addingTimeInterval(60*60))
task.cancel()
exit(0)
