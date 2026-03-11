import serial

class _Motor:
    """motor that take 2 or 3 arguments: dx, dy, (dtrigger, optional) in mm
    """
    def __init__(self, serial_port='COM3', baud_rate=9600, timeout=60, stepper_ratio=200) -> None: 
        self.stepper_ratio = stepper_ratio
        self.serial_port = serial_port
        self.baud_rate = baud_rate
        self.timeout = timeout
        
        self.serial_device = serial.Serial(port=serial_port, baudrate=baud_rate, timeout=timeout)
    
    def __exit__(self):
        self.serial_device.close()


    def reconnect(self):
        self.serial_device.close()
        self.serial_device = serial.Serial(port=self.serial_port, baudrate=self.baud_rate, timeout=self.timeout)
    
    def move(self, *argv):
        '''
        dx, dy, (dtrigger, optional) in mm
        will be convert to steps and send to stepper motor controller (arduino)
        '''
        assert len(argv) == 2 or len(argv) == 3

        msg = ','.join([str(int(x*self.stepper_ratio)) for x in argv])
        self.serial_device.write(msg.encode())
        resp = self.serial_device.readline()

        return 


class Motor_2d(_Motor):
    """motor that use arduino_motor_driver.ino arduino script with 4 inputs (dx, dy, dz, dtrigger) 
    but only use 2-3 inputs (dx, dy, (dtrigger)) since dz is always 0
    """
    def move(self, *argv):
        '''
        dx, dy, (dtrigger, optional) in mm
        will be convert to steps and send to stepper motor controller (arduino)

        since using the arduino script that has 4 inputs (the one same as Motor_H_1D), d1d is always 0
        '''
        assert len(argv) == 2 or len(argv) == 3

        d1d = 0
        cmd = list(argv)[:2] + [d1d] + list(argv)[2:]
        msg = ','.join([str(int(x*self.stepper_ratio)) for x in cmd])
        self.serial_device.write(msg.encode())
        resp = self.serial_device.readline()
        return 
    

class Motor_3d(_Motor):
    """motor that use arduino_motor_driver.ino arduino script with 3-4 inputs (dx, dy, dz, (dtrigger)) 
    """
    def __init__(self, serial_port='COM3', baud_rate=9600, timeout=60, stepper_ratio=200):
        super().__init__(serial_port, baud_rate, timeout, stepper_ratio)
    
    def move(self, *argv):
        '''
        dx, dy, d1d, (dtrigger, optional) in mm
        will be convert to steps and send to stepper motor controller (arduino)

        d1d is the distance for 1d rail guide.
        '''
        assert len(argv) == 3 or len(argv) == 4

        msg = ','.join([str(int(x*self.stepper_ratio)) for x in argv])
        self.serial_device.write(msg.encode())
        resp = self.serial_device.readline()

        return 


# class Motor_1d_using_3d(Motor):
#     def move(self, *argv):
#         '''
#         dx, (dtrigger, optional) in mm
#         will be convert to steps and send to stepper motor controller (arduino)

#         since using the arduino script that has 4 inputs (the one same as Motor_H_1D), d1d is always 0
#         '''
#         assert len(argv) == 1 or len(argv) == 2

#         d1d = 0
#         dy = 0
#         cmd = list(argv)[:1] + [dy, d1d] + list(argv)[1:]
#         msg = ','.join([str(int(x*self.stepper_ratio)) for x in cmd])
#         self.serial_device.write(msg.encode())
#         resp = self.serial_device.readline()
#         return 