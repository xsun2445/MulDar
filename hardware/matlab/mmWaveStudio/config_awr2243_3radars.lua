-- sequence of tx radar
tx_seq = {0,1,2}

IDEL_TIME =             15.0
FREQ_SLOPE =            38.816929   -- MHz/us
ADC_START_TIME =        7           -- us
TX_START_TIME =         0           -- us
ADC_SAMPLES =           256         -- number of samples
ADC_SAMPLE_RATE =       2150        -- ksps
RAMP_END_TIME =         128.50      -- us
BACKOFF_TX1 = 0
BACKOFF_TX2 = 0
BACKOFF_TX3 = 0
PhaseShifter_TX1 = 0
PhaseShifter_TX2 = 0
PhaseShifter_TX3 = 0

-- Bi-static chirp profile
START_FREQ =            76.01       -- GHz
-- -- or use the triggering time difference (coarser)
-- t_delta =               0           -- us (currently always 0 for the different starting frequency)



function bi_static_chirp_profile(profile_id, freq)
--[[
Configuring the chirp profile for bi-static radars
isTx: true for transmitting and receiving, false for only receiving
]]--

    -- Int32 ar1.ProfileConfig(UInt16 profileId, Single startFreqConst, Single idleTimeConst, Single adcStartTimeConst, Single rampEndTime, UInt32 tx0OutPowerBackoffCode, UInt32 tx1OutPowerBackoffCode, UInt32 tx2OutPowerBackoffCode, Single tx0PhaseShifter, Single tx1PhaseShifter, Single tx2PhaseShifter, Single freqSlopeConst, Single txStartTime, UInt16 numAdcSamples, UInt16 digOutSampleRate, UInt32 hpfCornerFreq1, UInt32 hpfCornerFreq2, Char rxGain) - Profile configuration API which defines chirp profile parameters
    -- _I_ UInt16	profileId	 - Chirp Profile Id [0 to 3]
    -- _I_ Single	startFreqConst	 - Chirp Start Frequency in GHz
    -- _I_ Single	idleTimeConst	 - Chirp Idle Time in µs
    -- _I_ Single	adcStartTimeConst	 - Chirp ADC Start Time in µs
    -- _I_ Single	rampEndTime	 - Chirp Ramp End Time in µs
    -- _I_ UInt32	tx0OutPowerBackoffCode	 - TX0 channel Power Backoff in dB
    -- _I_ UInt32	tx1OutPowerBackoffCode	 - TX1 channel Power Backoff in dB
    -- _I_ UInt32	tx2OutPowerBackoffCode	 - TX2 channel Power Backoff in dB
    -- _I_ Single	tx0PhaseShifter	 - TX0 channel Phase Shifter Value in deg
    -- _I_ Single	tx1PhaseShifter	 - TX1 channel Phase Shifter in deg
    -- _I_ Single	tx2PhaseShifter	 - TX2 channel Phase Shifter in deg
    -- _I_ Single	freqSlopeConst	 - Chirp Frequency Slope in MHz/µs
    -- _I_ Single	txStartTime	 - TX Start Time in µs
    -- _I_ UInt16	numAdcSamples	 - RX Number of Adc Samples
    -- _I_ UInt16	digOutSampleRate	 - RX Sampling Rate in ksps
    -- _I_ UInt32	hpfCornerFreq1	 - RX HPF1 corner frequency,[b15:0 (0x00-175 kHz, 0x01-235 kHz, 0x02-350 kHz, 0x03-700 kHz)] 
    --                                      + TxChnCalibSet[b31:16]
    -- _I_ UInt32	hpfCornerFreq2	 - RX HPF2 corner frequency,[b15:0 (0x00-350 kHz, 0x01-700 kHz, 0x02-1.4 MHz, 0x03-2.8 MHz)] 
    --                                      + ForceVCOSelet[b16] and VCOSelect[b17] , RetainTxCalUpdate[b24] , RetainRxCalLut[b25]
    -- _I_ Char	rxGain	 - RX Gain in dB(b0:5), RF Gain Target(b6:7)values 30dB:00, 33dB:01, 36dB:10, Reserved:11
    if (0 == ar1.ProfileConfig(
        profile_id, 
        freq, 
        IDEL_TIME, 
        ADC_START_TIME, 
        RAMP_END_TIME, 
        BACKOFF_TX1, 
        BACKOFF_TX2, 
        BACKOFF_TX3, 
        PhaseShifter_TX1, 
        PhaseShifter_TX2, 
        PhaseShifter_TX3, 
        FREQ_SLOPE, 
        TX_START_TIME, 
        ADC_SAMPLES, 
        ADC_SAMPLE_RATE, 
        0, 131072, 94)) then
        WriteToLog("ProfileConfig Success\n", "green")
    else
        WriteToLog("ProfileConfig Failure\n", "red")
    end
end


-- setting tx and rx profile
start_freq = START_FREQ
bi_static_chirp_profile(0, start_freq)
-- radar 0 redceiving from radar 1
f_delta = 0.6  -- MHz
start_freq = START_FREQ + f_delta*1e-3
bi_static_chirp_profile(1, start_freq)
-- radar 0 redceiving from radar 2
f_delta = 0.4  -- MHz
start_freq = START_FREQ + f_delta*1e-3
bi_static_chirp_profile(2, start_freq)


-- Chirp sequence
for i = 1,#tx_seq do
    curr_profile_id = i-1
    curr_chirp_idx = 2*(i-1)
    -- -- Int32 ar1.ChirpConfig(UInt16 chirpStartIdx, UInt16 chirpEndIdx, UInt16 profileId, Single startFreqVar, Single freqSlopeVar, Single idleTimeVar, Single adcStartTimeVar, UInt16 tx0Enable, UInt16 tx1Enable, UInt16 tx2Enable) - Chirp configuration API which defines which profile is to be used for each chirp in a frame
    -- -- _I_ UInt16	chirpStartIdx	 - First Chirp Start Index number
    -- -- _I_ UInt16	chirpEndIdx	 - Last chirp Index number
    -- -- _I_ UInt16	profileId	 - Chirp Configured profileId
    -- -- _I_ Single	startFreqVar	 - Chirp start frequency var in MHz
    -- -- _I_ Single	freqSlopeVar	 - frequency Slope Var in MHz/µs
    -- -- _I_ Single	idleTimeVar	 - Idle Time Var in µs
    -- -- _I_ Single	adcStartTimeVar	 - ADC Start Time Var in µs
    -- -- _I_ UInt16	tx0Enable	 - tx0 channel
    -- -- _I_ UInt16	tx1Enable	 - tx1 channel
    -- -- _I_ UInt16	tx2Enable	 - tx2 channel
    if tx_seq[i] == RADAR_IDX then
        ar1.ChirpConfig(curr_chirp_idx,   curr_chirp_idx,   curr_profile_id, 0, 0, 0, 0, 1, 0, 0)
        ar1.ChirpConfig(curr_chirp_idx+1, curr_chirp_idx+1, curr_profile_id, 0, 0, 0, 0, 0, 0, 1)
    else
        ar1.ChirpConfig(curr_chirp_idx,   curr_chirp_idx,   curr_profile_id, 0, 0, 0, 0, 0, 0, 0)
        ar1.ChirpConfig(curr_chirp_idx+1, curr_chirp_idx+1, curr_profile_id, 0, 0, 0, 0, 0, 0, 0)
    end
end

-- Frame Configs
NUM_FRAMES =        0   -- 0 for infinite frames
t_frame =           10
num_loop_chirp =    1   -- should always be 1 and only change num_frames for repeating chirps
num_chirps =        2*#tx_seq-1

-- Int32 ar1.FrameConfig(UInt16 chirpStartIdx, UInt16 chirpEndIdx, UInt16 frameCount, UInt16 loopCount, Single periodicity, Single triggerDelay, UInt16 TriggerSelect) - Frame Configuration API defines Frame formation which has sequence of chirps to be transmitted subsequently
-- _I_ UInt16	chirpStartIdx	 - First Chirp Start Index number
-- _I_ UInt16	chirpEndIdx	 - Last chirp Index number
-- _I_ UInt16	frameCount	 - Number of frames to transmit
-- _I_ UInt16	loopCount	 - Number of times to repeat from start chirp to last chirp in each frame
-- _I_ Single	periodicity	 - Each frame repetition period in ms
-- _I_ Single	triggerDelay	 -  Optional time delay from sync in trigger to the occurrence of frame chirps in µs
-- _I_ UInt16	TriggerSelect	 - TriggerSelect
if (0 == ar1.FrameConfig(0, num_chirps, NUM_FRAMES, num_loop_chirp, t_frame, 0, 2)) then
    WriteToLog("FrameConfig Success\n", "green")
else
    WriteToLog("FrameConfig Failure\n", "red")
end

ar1.StartFrame()