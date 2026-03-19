--Start
WriteToLog("LUA Script for System Check\n", "blue")
-- RSTD.Sleep(1000)
-- RSTD_PATH = RSTD.GetRstdPath()

if (0 == ar1.SOPControl(2)) then
    WriteToLog("SOP Reset Success\n", "green")
else
    WriteToLog("SOP Reset Failure\n", "red")
end

-- RSTD.Sleep(1000)
-- Connect usb to both capture card and radar
-- Capture card will show 4*AR-DevPack-Evm, radar will show 2*XDS110
-- choose the port XDS110 Class Application/User UART
if (0 == ar1.Connect(COM_PORT,115200,1000)) then
    WriteToLog("RS232 Connect Success\n", "green")
else
    WriteToLog("RS232 Connect Failure\n", "red")
end

-- RSTD.Sleep(1000)
if (0 == ar1.DownloadBSSFw(BSS_PATH)) then
    WriteToLog("BSS FW Download Success\n", "green")
else
    WriteToLog("BSS FW Download Failure\n", "red")
end

-- RSTD.Sleep(1000)

if (0 == ar1.DownloadMSSFw(MSS_PATH)) then
    WriteToLog("MSS FW Download Success\n", "green")
else
    WriteToLog("MSS FW Download Failure\n", "red")
end

-- RSTD.Sleep(1000)

if (0 == ar1.PowerOn(0, 1000, 0, 0)) then
    WriteToLog("PowerOn Success\n", "green")
else
    WriteToLog("PowerOn Failure\n", "red")
    session:destroy();
end

-- RSTD.Sleep(1000)

if (0 == ar1.RfEnable()) then
    WriteToLog("RfEnable Success\n", "green")
else
    WriteToLog("RfEnable Failure\n", "red")
end

-- RSTD.Sleep(1000)

-- -- Connect to DCA1000
-- ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098)
-- ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30)
-- ar1.CaptureCardConfig_PacketDelay(25)
-- -- [17:36:28]  [RadarAPI]: Sending fpga command to DCA1000
-- -- [17:36:28]  [RadarAPI]: 
-- -- [17:36:28]  FPGA Configuration command : Success
-- -- [17:36:28]  [RadarAPI]: Sending record command to DCA1000
-- -- [17:36:28]  [RadarAPI]: 
-- -- [17:36:28]  Configure Record command : Success
-- ar1.GetCaptureCardFPGAVersion()
-- -- [17:36:28]  [RadarAPI]: Sending fpga_version command to DCA1000
-- -- [17:36:29]  [RadarAPI]: 
-- -- [17:36:29]  
-- -- [17:36:29]  FPGA Version : 2.9 [Record]
-- -- [17:36:29]  
-- ar1.SelectCaptureDevice("DCA1000")
-- -- [17:36:23]  [RadarAPI]: Status: Passed
-- ar1.CaptureCard_DisConnect()


-- Data Path

-- StaticConfig

-- args: TX0-4, 
--       RX0-4, 
--       ADC bits, 
-- 		 full scale reduction factor, 
-- 		 IQ swap, 0:I first
if (0 == ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)) then
    WriteToLog("ChanNAdcConfig Success\n", "green")
else
    WriteToLog("ChanNAdcConfig Failure\n", "red")
end
-- ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 393217, 2, 1, 0)
-- ar1.ChanNAdcConfig(1, 1, 1, 1, 1, 1, 1, 2, 1, 0)
-- ar1.ChanNAdcConfig(1, 1, 0, 1, 1, 1, 1, 2, 1, 0)

-- ar1.RfLdoBypassConfig(0x1)
-- 0x1 bypass enable
-- 0x0 bypass disable
-- if (0 == ar1.RfLdoBypassConfig(0x0)) then
if (0 == ar1.RfLdoBypassConfig(0x1)) then
	WriteToLog("RfLdoBypassConfig Success\n", "green")
else
	WriteToLog("RfLdoBypassConfig Failure\n", "red")
end

if (0 == ar1.LPModConfig(0, 0)) then
    WriteToLog("LowPowerADCConfig Success\n", "green")
else
    WriteToLog("LowPowerADCConfig Failure\n", "red")
end

-- if (0 == ar1.SetMiscConfig(0, 0, 0, 0, 0, 0, 0, 0.4)) then
-- 	WriteToLog("SetMiacConfig Success\n", "green")
-- else
-- 	WriteToLog("SetMiacConfig Failure\n", "red")	
-- end


if (0 == ar1.SetCalMonFreqLimitConfig(76,81)) then
    WriteToLog("RfSetCalMonFreqTxPowLimitConfig Success\n", "green")
else
    WriteToLog("RfSetCalMonFreqTxPowLimitConfig Failure\n", "red")
end


if (0 == ar1.RfSetCalMonFreqTxPowLimitConfig(76, 76, 76, 81, 81, 81, 0, 0, 0)) then
    WriteToLog("RfSetCalMonFreqTxPowLimitConfig Success\n", "green")
else
    WriteToLog("RfSetCalMonFreqTxPowLimitConfig Failure\n", "red")
end


------------------------------------------------
ar1.SetApllSynthBWCtlConfig(0, 8, 38, 9)
-- ar1.SetMiscConfig(1, 1, 1, 0)
------------------------------------------------


if (0 == ar1.RfInit()) then
    WriteToLog("RfInit Success\n", "green")
else
    WriteToLog("RfInit Failure\n", "red")
end

-- DataConfig

if (0 == ar1.DataPathConfig(513, 1216644097, 0)) then
    WriteToLog("DataPathConfig Success\n", "green")
else
    WriteToLog("DataPathConfig Failure\n", "red")
end

if (0 == ar1.LvdsClkConfig(1, 1)) then
    WriteToLog("LvdsClkConfig Success\n", "green")
else
    WriteToLog("LvdsClkConfig Failure\n", "red")
end

ar1.DisableTestSource(0)

if (0 == ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0)) then   
    WriteToLog("LVDSLaneConfig Success\n", "green")
else
    WriteToLog("LVDSLaneConfig Failure\n", "red")
end

if (0 == ar1.SetDynamicPowerSaveMode(0, 0, 0)) then   
    WriteToLog("SetDynamicPowerSaveMode Success\n", "green")
else
    WriteToLog("SetDynamicPowerSaveMode Failure\n", "red")
end