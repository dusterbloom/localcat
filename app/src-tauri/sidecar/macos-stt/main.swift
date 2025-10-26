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

// PROPER FIX: Check authorization status without requesting it
// The main Tauri app should handle authorization, subprocess just checks status
if #available(macOS 10.15, *) {
    let status = SFSpeechRecognizer.authorizationStatus()
    if status == .notDetermined {
        // If not determined, the main app needs to request authorization first
        // For now, try to continue - some environments allow on-device without authorization
        fputs("{\"info\":\"authorization_not_determined\"}\n", stderr)
    } else if status == .denied {
        fputs("{\"error\":\"speech_denied\"}\n", stderr)
        exit(3)
    } else if status == .restricted {
        fputs("{\"error\":\"speech_restricted\"}\n", stderr)
        exit(4)
    }
    // If authorized, proceed normally
}

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

        // Distinguish between volatile (interim) and finalized results
        let obj: [String: Any]
        if final {
            obj = [
                "text": text,
                "final": true,
                "type": "finalized"
            ] as [String : Any]
        } else {
            obj = [
                "text": text,
                "final": false,
                "type": "volatile"
            ] as [String : Any]
        }

        if let data = try? JSONSerialization.data(withJSONObject: obj, options: []) {
            out.write(data)
            out.write("\n".data(using: .utf8)!)
        }

        if final { finished = true }
    } else if let error = error {
        let obj = ["error": String(describing: error), "type": "error"]
        if let data = try? JSONSerialization.data(withJSONObject: obj, options: []) {
            out.write(data); out.write("\n".data(using: .utf8)!)
        }
    }
}

// Read stdin continuously in chunks and stream to recognition
queue.async {
    let stdinFH = FileHandle.standardInput
    let chunkSize = 8192  // Read 8KB chunks at a time

    while true {
        // Read next chunk from stdin
        guard let chunk = try? stdinFH.read(upToCount: chunkSize) else {
            // Error reading stdin, end recognition
            request.endAudio()
            break
        }

        if chunk.isEmpty {
            // EOF reached, end recognition
            request.endAudio()
            break
        }

        // Convert chunk to audio buffer
        let frameCount = chunk.count / 2
        guard frameCount > 0 else { continue }

        guard let buf = AVAudioPCMBuffer(pcmFormat: fmt, frameCapacity: AVAudioFrameCount(frameCount)) else {
            fputs("{\"error\":\"invalid_audio_format\"}\n", stderr)
            continue
        }

        buf.frameLength = AVAudioFrameCount(frameCount)
        chunk.withUnsafeBytes { rawPtr in
            guard let base = rawPtr.bindMemory(to: Int16.self).baseAddress else { return }
            buf.int16ChannelData!.pointee.update(from: base, count: frameCount)
        }

        // Append this chunk to the ongoing recognition request
        request.append(buf)
    }
}

// Keep runloop alive
RunLoop.current.run(until: Date().addingTimeInterval(60*60))
task.cancel()
exit(0)
