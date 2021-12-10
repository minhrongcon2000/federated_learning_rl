# version 91 changed from v9, calculate sleeptime using formula 
# and add  SLEEP_FINISH event after calculated sleeptime
import math
import numpy as np
from video_streaming_env.singlepath_env.envs.utils import VideoListCollector
from video_streaming_env.singlepath_env.envs.utils import Event

# from utils import DownloadPath
import gym


class SinglepathEnvGym(gym.Env):
    EVENT = Event
    SAMPLE = 0.05  # Second, check event every after SAMPLE time
    NETWORK_SEGMENT = 1  # Second

    BUFFER_NORM_FACTOR = 10
    REWARD_PENALTY = 1
    SMOOTH_PENALTY = 1

    NETWORK_SPEED_NORM = 1e6
    CHUNK_STATE_NORM = 0.1

    DEFAULT_QUALITY = 0  # default video quality without agent
    MIN_BUFFER_THRESH = 6.0  # sec, min buffer threshold

    # video bitrate is used as a ultility reward for each bitrate level so this can be change however fit
    # Youtube recommend
    UTILITY_SCORE = np.array(
        [700, 900, 2000, 3000, 5000, 6000, 8000]
    )  # for 04-second segment
    VIDEO_BIT_RATE = np.array(
        [700, 900, 2000, 3000, 5000, 6000, 8000]
    )  # for 04-second segmen

    QUALITY_SPACE = 7  # Number of quality levels
    HISTORY_SIZE = 6  # Number of network speed size to keep
    CHUNK_TIL_VIDEO_END = 60  # Number of chunks in a video
    BUFFER_THRESHOLD = 30  # Max buffer
    M_IN_K = 1000
    VIDEO_CHUNK_LEN = 4  # Length of each chunk, in second

    # rules = ["duplicate", "no-duplicate", "greedy"]

    def __init__(self, log_qoe=True, bitrate_list=None, replace=True, train=True):
        if log_qoe:
            self.UTILITY_SCORE = np.log(self.UTILITY_SCORE / self.UTILITY_SCORE[0])
            self.REBUF_PENALTY = 2.66  # 1 sec rebuffering -> 3 Mbps
        else:
            self.REBUF_PENALTY = 4.3
        self.bitrate_list = bitrate_list
        self.replace = replace
        self.train = train
        self.eps_counter = 0
        
        # Get video list
        # Return is a matrix with shape (QUALITY_LEVEL, NUMBER_OF_CHUNKS)
        # Entry[i, j] represents bitrate of chunk j at quality level i
        self.video_list = VideoListCollector().get_trace_matrix(self.VIDEO_BIT_RATE)

        self.action_space = gym.spaces.Discrete(self.QUALITY_SPACE)
        obs_space = dict(
            network_speed=gym.spaces.Box(
                low=0, high=float("inf"), shape=(self.HISTORY_SIZE,), dtype=np.float32
            ),
            next_chunk_size=gym.spaces.Box(
                low=0, high=float("inf"), shape=(self.QUALITY_SPACE,), dtype=np.float32
            ),
            buffer_size=gym.spaces.Box(
                low=0, high=float("inf"), shape=(1,), dtype=np.float32
            ),
            percentage_remain_video_chunks=gym.spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
            last_down_quality=gym.spaces.Discrete(7),
            delay=gym.spaces.Box(
                low=0, high=float("inf"), shape=(self.HISTORY_SIZE,), dtype=np.float32
            ),
        )
        self.observation_space = gym.spaces.Dict(obs_space)

    def pick_next_segment(self):
        for i in range(self.CHUNK_TIL_VIDEO_END):
            if self.download_segment[i] == -1:
                return i

    def reset(self):
        # Inputs as a tuple of two
        if self.train:
            bw_idx = np.random.choice(len(self.bitrate_list), replace=self.replace)
            self.bw = self.bitrate_list[bw_idx, :]
            self.init_net_seg = np.random.randint(0, len(self.bw) - 1)
        else:
            self.bw = self.bitrate_list[self.eps_counter, :]
            self.eps_counter += 1
            self.eps_counter = self.eps_counter % len(self.bitrate_list)
            self.init_net_seg = 0

        self.RTT = np.random.uniform(0.05, 0.1)  # Round Trip Time
        self.play_id = 0
        self.download_segment = np.array([-1] * self.CHUNK_TIL_VIDEO_END)
        self.download_segment_finish = np.array([-1] * self.CHUNK_TIL_VIDEO_END)

        self.reward_quality = 0.0  # reward of episode
        self.reward_smooth = 0.0  # reward of episode
        self.rebuffer_time_simu = (
            0.0  # rebuffering time for episode, calculated from simulation
        )
        self.rebuffer_time = (
            0.0  # rebuffering time for episode, calculated from formula
        )

        self.est_throughput = 0.0

        self.total_reward = 0.0  # reward of episode
        self.reward_quality_norm = (
            0.0  # reward_quality of episode, normalized by M_IN_K
        )
        self.reward_smooth_norm = 0.0  # smooth reward of episode, normalied by M_IN_K
        self.reward_rebuffering_norm = (
            0.0  # rebuffering time reward of episode, normalized by M_IN_K
        )
        self.num_switch = 0  # counter of number of quality switch

        self.end_of_video = False
        self.buffer_size_trace = 0.0
        self.sleep_time = 0.0
        # Event format: [[timestamp, event, down_id, down_quality]]

        # First down events
        self.event = np.array([[0.0, self.EVENT.DOWN, 0, self.DEFAULT_QUALITY]])
        self.download_segment[0] = self.DEFAULT_QUALITY

        segment_size = self.video_list[0][0] * 8
        delay = self.down_time(segment_size, 0.0)

        self.event = np.concatenate(
            (self.event, [[delay, self.EVENT.DOWN_FINISH, 0, self.DEFAULT_QUALITY]])
        )
        self.download_segment_finish[0] = self.DEFAULT_QUALITY
        self.buffer_size_trace += self.VIDEO_CHUNK_LEN

        # Add play event
        self.event = np.concatenate(
            (self.event, [[delay + 0.00001, self.EVENT.PLAY, 0, self.DEFAULT_QUALITY]])
        )

        self.event = self.event[self.event[:, 0].argsort()]  # Sort by cur_time
        self.event = np.delete(self.event, 0, 0)  # Remove first event

        # Log
        self.last_quality = self.DEFAULT_QUALITY
        self.network_speed = np.zeros(self.HISTORY_SIZE)  # Estimate network speed
        self.delay_net = np.zeros(
            self.HISTORY_SIZE
        )  # Download time of previous 6 chunks, for this part
        self.state = dict(
            network_speed=self.network_speed,
            next_chunk_size=np.zeros(
                self.QUALITY_SPACE,
            ),
            buffer_size=np.array([0]),
            percentage_remain_video_chunks=np.array([1]),
            last_down_quality=self.DEFAULT_QUALITY,
            delay=self.delay_net,
        )
        return self.state

    def down_time(self, segment_size, cur_time):
        # calculate net_seg_id, seg_time_stamp from cur_time. Remember seg_time_stamp plus RTT
        # set network segment ID to position after sleeping and download last segment
        delay = self.RTT
        pass_seg = math.floor(cur_time / self.NETWORK_SEGMENT)
        net_seg_id = self.init_net_seg + pass_seg
        seg_time_stamp = cur_time - pass_seg

        while (
            True
        ):  # download segment process finish after a full video segment is downloaded
            net_seg_id = net_seg_id % len(self.bw)  # loop back to begin if finished
            network = self.bw[net_seg_id]  # network DL_bitrate in bps
            max_throughput = network * (self.NETWORK_SEGMENT - seg_time_stamp)

            if max_throughput > segment_size:  # finish download in network segment
                seg_time_stamp += (
                    segment_size / network
                )  # used time in network segment in second
                delay += segment_size / network  # delay from begin in second
                break
            else:
                delay += (
                    self.NETWORK_SEGMENT - seg_time_stamp
                )  # delay from begin in second
                seg_time_stamp = 0  # used time of next network segment is 0s
                segment_size -= (
                    max_throughput  # remain undownloaded part of video segment
                )
                net_seg_id += 1

        return delay

    def step(self, action):
        down_id = self.pick_next_segment()
        down_quality = action

        # NEW STEP
        cur_down_time = self.event[0][0]  # Current event time
        chunk_sleep_time = 0.0

        self.download_segment[down_id] = down_quality

        # video list is matrix of shape (7, num_chunks)
        segment_size = (
            float(self.video_list[down_quality][down_id]) * 8.0
        )  # download video segment in bits
        delay = self.down_time(segment_size, cur_down_time)
        chunk_rebuffer_time = max(0, delay - self.buffer_size_trace)

        self.event = np.concatenate(
            (
                self.event,
                [
                    [
                        cur_down_time + delay,
                        self.EVENT.DOWN_FINISH,
                        down_id,
                        down_quality,
                    ]
                ],
            )
        )
        self.event = np.delete(
            self.event, 0, 0
        )  # remove the current considering event from event

        self.event = self.event[self.event[:, 0].argsort()]  # Sort the event by time

        while True:
            cur_time = self.event[0][0]  # Earliest time

            # print(cur_time, self.event[0][1])

            if self.event[0][1] == self.EVENT.DOWN:
                if self.pick_next_segment() is None:
                    self.event = np.concatenate(
                        (
                            self.event,
                            [
                                [
                                    cur_time + self.buffer_size_trace,
                                    self.EVENT.SLEEP_FINISH,
                                    self.buffer_size_trace,
                                    -1,
                                ]
                            ],
                        )
                    )
                    self.event = np.delete(self.event, 0, 0)
                else:
                    break

            if self.event[0][1] == self.EVENT.DOWN_FINISH:
                self.download_segment_finish[self.event[0][2]] = self.event[0][3]

                self.buffer_size_trace = (
                    max(0, self.buffer_size_trace - delay) + self.VIDEO_CHUNK_LEN
                )

                if self.buffer_size_trace > self.BUFFER_THRESHOLD:
                    sleep_period = self.buffer_size_trace - self.BUFFER_THRESHOLD
                    self.event = np.concatenate(
                        (
                            self.event,
                            [
                                [
                                    cur_time + sleep_period,
                                    self.EVENT.SLEEP_FINISH,
                                    sleep_period,
                                    -1,
                                ]
                            ],
                        )
                    )
                else:
                    self.event = np.concatenate(
                        (self.event, [[cur_time + 0.00001, self.EVENT.DOWN, -1, -1]])
                    )

            if (
                self.event[0][1] == self.EVENT.SLEEP_FINISH
            ):  # this make an infinite loop
                chunk_sleep_time += self.event[0][2]
                self.buffer_size_trace = max(
                    0, self.buffer_size_trace - self.event[0][2]
                )

                if self.buffer_size_trace > self.BUFFER_THRESHOLD:
                    sleep_period = self.buffer_size_trace - self.BUFFER_THRESHOLD
                    self.event = np.concatenate(
                        (
                            self.event,
                            [
                                [
                                    cur_time + sleep_period,
                                    self.EVENT.SLEEP_FINISH,
                                    sleep_period,
                                    -1,
                                ]
                            ],
                        )
                    )
                else:
                    self.event = np.concatenate(
                        (self.event, [[cur_time + 0.00001, self.EVENT.DOWN, -1, -1]])
                    )

            if self.event[0][1] == self.EVENT.PLAY:
                self.play_id = int(self.event[0][2])
                play_quality = self.download_segment_finish[self.play_id]

                self.event = np.concatenate(
                    (
                        self.event,
                        [
                            [
                                cur_time + self.VIDEO_CHUNK_LEN,
                                self.EVENT.PLAY_FINISH,
                                self.play_id,
                                play_quality,
                            ]
                        ],
                    )
                )

            if self.event[0][1] == self.EVENT.PLAY_FINISH:
                self.play_id = int(self.event[0][2])  # finish play_id

                if self.play_id == self.CHUNK_TIL_VIDEO_END - 1:
                    self.event = np.delete(self.event, 0, 0)
                    break

                if self.download_segment_finish[self.play_id + 1] == -1:
                    self.event = np.concatenate(
                        (
                            self.event,
                            [
                                [
                                    cur_time + self.SAMPLE,
                                    self.EVENT.FREEZE_FINISH,
                                    self.play_id + 1,
                                    -1,
                                ]
                            ],
                        )
                    )  # waiting for play_id+1 chunk
                else:
                    self.event = np.concatenate(
                        (
                            self.event,
                            [
                                [
                                    cur_time + 0.00001,
                                    self.EVENT.PLAY,
                                    self.play_id + 1,
                                    self.download_segment_finish[self.play_id + 1],
                                ]
                            ],
                        )
                    )

            if self.event[0][1] == self.EVENT.FREEZE_FINISH:
                self.rebuffer_time_simu += self.SAMPLE
                if (
                    self.download_segment_finish[int(self.event[0][2])] == -1
                ):  # next chunk has not downloaded yet
                    self.event = np.concatenate(
                        (
                            self.event,
                            [
                                [
                                    cur_time + self.SAMPLE,
                                    self.EVENT.FREEZE_FINISH,
                                    self.event[0][2],
                                    -1,
                                ]
                            ],
                        )
                    )  # waiting for event[0][2]
                else:
                    self.event = np.concatenate(
                        (
                            self.event,
                            [
                                [
                                    cur_time + 0.00001,
                                    self.EVENT.PLAY,
                                    self.event[0][2],
                                    self.download_segment_finish[int(self.event[0][3])],
                                ]
                            ],
                        )
                    )

            self.event = np.delete(
                self.event, 0, 0
            )  # remove the current considering event from event
            self.event = self.event[self.event[:, 0].argsort()]

        # calculate QoE metrics
        self.sleep_time += chunk_sleep_time

        down_quality = self.download_segment[down_id]
        last_down_quality = self.download_segment[down_id - 1]
        chunk_reward_quality = self.UTILITY_SCORE[down_quality]
        chunk_reward_smooth = np.abs(
            self.UTILITY_SCORE[down_quality] - self.UTILITY_SCORE[last_down_quality]
        )
        if down_quality != last_down_quality:
            self.num_switch += 1

        chunk_reward_norm = (
            chunk_reward_quality * self.REWARD_PENALTY / 100
            - chunk_reward_smooth * self.SMOOTH_PENALTY / 100
            - chunk_rebuffer_time * self.REBUF_PENALTY / 100
        )

        self.reward_quality += chunk_reward_quality
        self.reward_quality_norm = self.reward_quality * self.REWARD_PENALTY / 100

        self.reward_smooth += chunk_reward_smooth
        self.reward_smooth_norm = self.reward_smooth * self.SMOOTH_PENALTY / 100

        self.rebuffer_time += chunk_rebuffer_time  # cumulative rebuffering time
        self.reward_rebuffering_norm = self.rebuffer_time * self.REBUF_PENALTY / 100

        self.total_reward = (
            self.reward_quality_norm
            - self.reward_smooth_norm
            - self.reward_rebuffering_norm
        )

        # Calculate new state
        next_chunk_size = (
            self.video_list[:, down_id + 1]
            if down_id + 1 < self.CHUNK_TIL_VIDEO_END
            else np.array([0] * self.QUALITY_SPACE)
        )

        self.est_throughput = segment_size / delay  # in bits per second
        self.network_speed = np.roll(self.network_speed, axis=-1, shift=1)
        self.network_speed[0] = self.est_throughput / self.NETWORK_SPEED_NORM
        self.delay_net = np.roll(self.delay_net, axis=-1, shift=1)
        self.delay_net[0] = delay

        remain = self.CHUNK_TIL_VIDEO_END - down_id
        self.state = dict(
            network_speed=self.network_speed,
            next_chunk_size=next_chunk_size / self.NETWORK_SPEED_NORM,
            buffer_size=np.array([self.buffer_size_trace / self.BUFFER_NORM_FACTOR]),
            percentage_remain_video_chunks=np.array(
                [remain / self.CHUNK_TIL_VIDEO_END]
            ),
            last_down_quality=int(self.download_segment[down_id]),
            delay=self.delay_net,
        )
        if self.play_id == self.CHUNK_TIL_VIDEO_END - 1:
            self.end_of_video = True

        info = dict(
            down_id=down_id,
            sum_reward=self.total_reward,
            download_segment=self.download_segment,
            delay=delay,
            reward_quality_norm=self.reward_quality_norm,
            reward_smooth_norm=self.reward_smooth_norm,
            reward_rebuffering_norm=self.reward_rebuffering_norm,
        )
        return self.state, chunk_reward_norm, self.end_of_video, info
    