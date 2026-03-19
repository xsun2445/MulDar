import argparse
import socket
import time
from typing import Callable
import wiringpi
import RPi.GPIO as GPIO


class Trigger:
    """Encapsulates Raspberry Pi GPIO pulse generation.

    Uses RPi.GPIO if available; otherwise falls back to a no-op simulator.
    """

    def __init__(self) -> None:
        wiringpi.wiringPiSetupGpio()
        # output
        for i in [27, 22, 23, 24, 25]:
            wiringpi.pinMode(i,1)
        # input
        for i in [26]:
            wiringpi.pinMode(i,0)

    def sendPulse(self, num_loop, pd_ms):
        """C implementation of simultaneously controlling multiple gpio in us."""
        wiringpi.sendPulseToRadar2(0xff, num_loop, int(pd_ms*1e3), int(1))

    def motor_trigger(self, num_loop, pd_ms):
        import time
        thresh_t = 0.001
        prev_t = 0
        prev_t = time.time()
        cnt = 0

        while True:
            while wiringpi.digitalRead(26) == 0 or time.time()-prev_t < thresh_t:
                continue

            self.sendPulse(num_loop, pd_ms)
            cnt += 1
            if cnt % 100 == 0:
                print(time.time()-prev_t, '\t', cnt)
            prev_t = time.time()


def run_trigger_server(bind_ip: str,
                       port: int,
                       delimiter: str,
                       greeting: str,
                       on_trigger: Callable[[int, float], None]) -> None:
    """Raspberry Pi TCP server that waits for connections and triggers on demand.

    - Listens on (bind_ip, port).
    - For each connection: expects a message f"{greeting}{delimiter}{num_frames}{delimiter}{period_ms}".
    - Runs on_trigger(num_frames, period_ms), then replies with b"Done.", and closes.
    - Keeps running to accept the next connection.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((bind_ip, port))
    srv.listen(1)
    print(f"Trigger server listening on {bind_ip}:{port}")

    while True:
        conn, addr = srv.accept()
        print(f"Client connected from {addr}")
        with conn:
            try:
                conn.settimeout(10)
                data = conn.recv(256)
                if not data:
                    print("Empty message; closing connection")
                    continue
                try:
                    text = data.decode(errors='ignore')
                except Exception:
                    text = ''

                parts = text.split(delimiter)
                if len(parts) >= 3 and parts[0] == greeting:
                    try:
                        num_frames = int(parts[1])
                        period_ms = int(parts[2])
                        on_trigger(num_frames, period_ms)
                        try:
                            conn.sendall(b'Done.')
                        except Exception:
                            pass
                    except ValueError:
                        try:
                            conn.sendall(b'Bad command')
                        except Exception:
                            pass
                else:
                    try:
                        conn.sendall(b'Bad command')
                    except Exception:
                        pass
            except Exception as e:
                print(f"Error handling client: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(description='Raspberry Pi trigger server (standby)')
    parser.add_argument('--bind-ip', default='0.0.0.0', help='Bind IP (use 0.0.0.0 to listen on all)')
    parser.add_argument('--port', type=int, default=5000, help='TCP port to listen on')
    parser.add_argument('--delimiter', default='@$@$', help='Message delimiter')
    parser.add_argument('--greeting', default='hello from server!', help='Expected greeting prefix')
    args = parser.parse_args()

    raspi = Trigger()
    run_trigger_server(
        bind_ip=args.bind_ip,
        port=args.port,
        delimiter=args.delimiter,
        greeting=args.greeting,
        on_trigger=raspi.sendPulse,
    )

if __name__ == '__main__':
    main()
