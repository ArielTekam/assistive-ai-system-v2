import time
import threading
import statistics

SYNC_SLEEP = 1.5
ASYNC_SLEEP = 1.5

NUM_ITERATIONS = 100

sync_fps = []
async_fps = []

print("\n==============================")
print("SYNC AUDIO TEST")
print("==============================")

start = time.time()

for i in range(NUM_ITERATIONS):

    frame_start = time.time()

    # simulation traitement vision
    time.sleep(0.03)

    # audio SYNCHRONE
    time.sleep(SYNC_SLEEP)

    frame_time = time.time() - frame_start

    fps = 1.0 / frame_time

    sync_fps.append(fps)

    print(f"[SYNC] Frame {i} | FPS:{fps:.2f}")

sync_total = time.time() - start

print("\n==============================")
print("ASYNC AUDIO TEST")
print("==============================")

def async_audio():
    time.sleep(ASYNC_SLEEP)

start = time.time()

for i in range(NUM_ITERATIONS):

    frame_start = time.time()

    # simulation vision
    time.sleep(0.03)

    # audio ASYNC
    threading.Thread(target=async_audio).start()

    frame_time = time.time() - frame_start

    fps = 1.0 / frame_time

    async_fps.append(fps)

    print(f"[ASYNC] Frame {i} | FPS:{fps:.2f}")

async_total = time.time() - start

print("\n==============================")
print("RESULTS")
print("==============================")

print(f"\nSYNC Mean FPS: {statistics.mean(sync_fps):.2f}")
print(f"SYNC Total Time: {sync_total:.2f}s")

print(f"\nASYNC Mean FPS: {statistics.mean(async_fps):.2f}")
print(f"ASYNC Total Time: {async_total:.2f}s")
