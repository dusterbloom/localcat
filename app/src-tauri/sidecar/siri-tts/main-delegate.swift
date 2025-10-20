// Siri TTS sidecar using delegate approach for reliability
import AVFoundation
import Accelerate

class SynthDelegate: NSObject, AVSpeechSynthesizerDelegate {
    var audioData = Data()
    var sampleRate: Double = 24000
    var channels: UInt16 = 1
    let sem = DispatchSemaphore(value: 0)

    func speechSynthesizer(_ synthesizer: AVSpeechSynthesizer, didFinish utterance: AVSpeechUtterance) {
        sem.signal()
    }
}

func writeWAVHeader(_ fh: FileHandle, sampleRate: Double, channels: UInt16, dataSize: UInt32) {
    func u32(_ v: UInt32) -> Data {
        var value = v.littleEndian
        return Data(bytes: &value, count: MemoryLayout<UInt32>.size)
    }
    func u16(_ v: UInt16) -> Data {
        var value = v.littleEndian
        return Data(bytes: &value, count: MemoryLayout<UInt16>.size)
    }

    var hdr = Data()
    hdr.append("RIFF".data(using: .ascii)!)
    hdr.append(u32(dataSize + 36))
    hdr.append("WAVEfmt ".data(using: .ascii)!)
    hdr.append(u32(16))                            // PCM fmt chunk size
    hdr.append(u16(1))                             // PCM
    hdr.append(u16(channels))                      // channels
    hdr.append(u32(UInt32(sampleRate)))            // sample rate
    hdr.append(u32(UInt32(sampleRate) * UInt32(channels) * 2)) // byte rate
    hdr.append(u16(channels * 2))                  // block align
    hdr.append(u16(16))                            // bits per sample
    hdr.append("data".data(using: .ascii)!)
    hdr.append(u32(dataSize))
    try! fh.write(contentsOf: hdr)
}

let args = CommandLine.arguments
guard args.count >= 3 else {
    fputs("Usage: siri-tts \"text\" out.wav [voice-id]\n", stderr)
    exit(2)
}

let text = args[1]
let outPath = args[2]
let voiceId = (args.count >= 4) ? args[3] : nil

let synth = AVSpeechSynthesizer()
let delegate = SynthDelegate()
synth.delegate = delegate

let utt = AVSpeechUtterance(string: text)

// Set voice
if let vid = voiceId, let v = AVSpeechSynthesisVoice(identifier: vid) {
    utt.voice = v
} else if let ava = AVSpeechSynthesisVoice(identifier: "com.apple.voice.enhanced.en-US.Ava") {
    utt.voice = ava
} else if let ava = AVSpeechSynthesisVoice(identifier: "com.apple.voice.compact.en-US.Ava") {
    utt.voice = ava
}

utt.rate = AVSpeechUtteranceDefaultSpeechRate
utt.pitchMultiplier = 1.0
utt.volume = 1.0

// Create temporary file for audio
let tempURL = URL(fileURLWithPath: "/tmp/siri-tts-\(ProcessInfo.processInfo.processIdentifier).caf")

do {
    // Use write() to get audio data
    synth.write(utt) { buffer in
        if let pcm = buffer as? AVAudioPCMBuffer {
            let fmt = pcm.format
            delegate.sampleRate = fmt.sampleRate
            delegate.channels = UInt16(fmt.channelCount)

            let frameLen = Int(pcm.frameLength)

            // Handle Int16
            if let ch0 = pcm.int16ChannelData?.pointee {
                let data = Data(bytes: ch0, count: frameLen * 2)
                delegate.audioData.append(data)
            }
            // Handle Float32 - convert to Int16
            else if let ch0f = pcm.floatChannelData?.pointee {
                var tmpFloat = [Float](repeating: 0, count: frameLen)
                var scale: Float = 32767.0
                vDSP_vsmul(ch0f, 1, &scale, &tmpFloat, 1, vDSP_Length(frameLen))

                let tmp = tmpFloat.map { Int16(max(-32768, min(32767, $0))) }
                let data = Data(bytes: tmp, count: frameLen * 2)
                delegate.audioData.append(data)
            }
        } else {
            // Completion - signal semaphore
            delegate.sem.signal()
        }
    }

    // Wait for synthesis to complete
    _ = delegate.sem.wait(timeout: .now() + 30)

    // Write WAV file
    guard FileManager.default.createFile(atPath: outPath, contents: nil),
          let fh = FileHandle(forWritingAtPath: outPath) else {
        fputs("Error: Cannot create output file: \(outPath)\n", stderr)
        exit(1)
    }

    writeWAVHeader(fh, sampleRate: delegate.sampleRate, channels: delegate.channels, dataSize: UInt32(delegate.audioData.count))
    try fh.write(contentsOf: delegate.audioData)
    try fh.close()

    print("ok \(outPath)")
} catch {
    fputs("Error: \(error)\n", stderr)
    exit(1)
}
