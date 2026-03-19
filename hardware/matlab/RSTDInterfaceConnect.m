function ErrStatus = RSTDInterfaceConnect()

addpath(genpath('.\'))

% Initialize mmWaveStudio .NET connection
RSTD_DLL_Path = 'C:\ti\mmwave_studio_03_00_00_14\mmWaveStudio\Clients\RtttNetClientController\RtttNetClientAPI.dll';

ErrStatus = Init_RSTD_Connection(RSTD_DLL_Path);
if (ErrStatus ~= 30000)
    disp('Error inside Init_RSTD_Connection');
    return;
end

% Lua_String = 'WriteToLog("Running script from MATLAB\n", "green")';
% 
% RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
% 
% % Example Lua Command
% % strFilename = 'C:\\ti\\mmwave_studio_01_00_00_01\\mmWaveStudio\\Scripts\\Example_script_AllDevices.lua';
% strFilename = 'C:\\Workspace\\SAR\\lua_scripts\\Trail_AWR2944.lua';
% Lua_String = sprintf('dofile("%s")',strFilename);
% % Lua_String_2 = sprintf('assert(loadfile("%s"))(10)',strFilename);
% ErrStatus =RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
% disp(ErrStatus)

end