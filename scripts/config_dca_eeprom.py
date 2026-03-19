# config DCA1000 device ip address by writing to EEPROM, 
# Allowing multiple radar data collection simultanously
import sys
import glob
import serial

from muldar.devices.radar import DCA1000


if __name__ == '__main__':
    # original and default ip for DCA1000
    sys_ip_ori      = '192.168.33.30'
    device_ip_ori   = '192.168.33.180'
    cfg_port_ori    = 4096
    data_port_ori   = 4098

    sys_ip_tar      = '192.168.33.30'
    device_ip_tar   = '192.168.33.181'
    cfg_port_tar    = 4200
    data_port_tar   = 4201
    mac_tar         = '12.34.56.78.91.01'

    # connectivity check
    dca = DCA1000(system_ip     = sys_ip_ori, 
                dca_ip          = device_ip_ori, 
                config_port     = cfg_port_ori, 
                data_port       = data_port_ori)
    dca.configure()
    dca.close()

    # config DCA1000 device ip 
    DCA1000.config_eeprom(
        dca_ip              = device_ip_ori,
        dca_port            = cfg_port_ori,
        system_ip           = sys_ip_ori,
        DCA1000IPAddress    = device_ip_tar,
        DCA1000MACAdress    = mac_tar,
        DCA1000ConfigPort   = cfg_port_tar, 
        DCA1000DataPort     = data_port_tar,
        systemIPAdress      = sys_ip_tar)

    # # connectivity check after config
    # dca = DCA1000(system_ip     = sys_ip_ori, 
    #             dca_ip          = device_ip_ori, 
    #             config_port     = cfg_port_ori, 
    #             data_port       = data_port_ori)
    # dca.configure()
    # dca.close()