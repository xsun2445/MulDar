% This script is used for retriggering the hardware trigger at each row
% during multiple radar collection.
% The main reason is to prevent the MSS async error on radar when the radar
% was triggered too long.

% addpath('utils')

%% configs

machineName = 'slave_6';

config_path = './configs/devices_comm_cfg.json';
devices_config = jsondecode(fileread(config_path)).comm_cfg;

COMMAND_LENGTH = devices_config.COMMAND_LENGTH;
DELIMITER = devices_config.DELIMITER;
wait_time = devices_config.CLIENT_MSG_WAIT_TIME;
reconnect_time = devices_config.CLIENT_RECONNECTION_TIME;

server_ip = devices_config.(machineName).IP;
server_port = devices_config.(machineName).server_access_port;


%% Connect to the mmWave Studio
for i = 1:3
    try
        if 30000 ~= RSTDInterfaceConnect()
            ME = MException('RSTD connection failed!');
            disp(ME)
            continue
            % throw(ME);
        end
        break
    catch
    end
end

%% MATLAB TCP Server (Non-blocking)
server = tcpserver(server_ip, server_port, "ConnectionChangedFcn", @onConnect);

disp("MATLAB TCP Server is running on port " + server_port);

function onConnect(server, event)
	if server.Connected
		msg = readline(server)
		if msg == "Retrigger"
    		disp("Received Retrigger!");
			retrigger();
		elseif contains(msg, "TriggerFrames") 
    		disp("Received TriggerFrames!");
			temp = split(msg, "@$@$");
			num_frame = str2num(temp(2));
			frame_td = str2num(temp(3));
			server.UserData.device.trigger_frame(num_frame,frame_td)
		elseif contains(msg, "ConfigRadar")
			disp("Received ConfigRadar!");
			temp = split(msg, "@$@$");
			start_freq = str2double(temp(2));
			start_freq = round(start_freq, 7)
			configradar(start_freq);
        elseif contains(msg, "RunGivenScript")
            disp("Received Script!")
            temp = split(msg, "@$@$");
            N = str2double(temp(2));
            t0 = tic;
            while server.NumBytesAvailable < N
                pause(0.01);
                if toc(t0) > 10
                    native2unicode(read(server, N, "uint8"), "UTF-8")
                    warning("Timed out for lua script");
                    return
                end
            end
            payload = read(server, N, "uint8");
            script_content = native2unicode(payload, "UTF-8");
            sendCmdStudio(script_content);
            disp("Successfully configured")

		end
	end
end


function retrigger()
	sendCmdStudio('ar1.StopFrame(4)');
	pause(1);
	sendCmdStudio('ar1.StartFrame()');
end


function configradar(newValue)
	
	filename = '';
	Lua_String = sprintf('dofile("%s")',filename)
	sendCmdStudio(Lua_String);

	filename = '';
	lines = readlines(filename);
	for i = 1:length(lines)
    	if contains(lines(i), 'START_FREQ = ')
        	lines(i) = "START_FREQ = " + num2str(newValue, '%.7f');
        	break;
    	end
	end
	allText = join(lines, newline);
	sendCmdStudio(allText);
end


function sendCmdStudio(Lua_String)
	ErrStatus = RtttNetClientAPI.RtttNetClient.SendCommand(Lua_String);
	if (ErrStatus ~= 30000)
    	disp('mmWaveStudio Connection Failed');
		fprintf([Lua_String, '\n']);
	end
end


