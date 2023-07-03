def pixel_metric(frame, prev_frame):
    return np.nanmean((frame.astype(np.int32) - prev_frame) ** 2)
def hist_metric(frame_hist,prev_frame_hist):
    return np.nanmean(abs(frame_hist - prev_frame_hist))
def hist(frame, mask, hist_size):
    return cv2.calcHist([frame],[0], mask, [hist_size], [0, 256])

def metric_generator(frames):
    metrics = []
    metric_values = []
    hist_metric_values = []
    hist_metric_values_1q = []
    hist_metric_values_2q = []
    hist_metric_values_3q = []
    hist_metric_values_4q = []
    hist_metric_values_5q = []
    hist_metric_values_6q = []
    hist_metric_values_7q = []
    hist_metric_values_8q = []
    hist_metric_values_9q = []

    ### START CODE HERE ###
    # Ваши внешние переменные

    mask = None
    mask_1q = None
    mask_2q = None
    mask_3q = None
    mask_4q = None
    mask_5q = None
    mask_6q = None
    mask_7q = None
    mask_8q = None
    mask_9q = None

    prev_frame = None
    prev_frame_hist = None
    prev_frame_1q_hist = None
    prev_frame_2q_hist = None
    prev_frame_3q_hist = None
    prev_frame_4q_hist = None
    prev_frame_5q_hist = None
    prev_frame_6q_hist = None
    prev_frame_7q_hist = None
    prev_frame_8q_hist = None
    prev_frame_9q_hist = None

    hist_size = 16
    frames_seq_len = 7

    ###  END CODE HERE  ###

    for idx, frame in tqdm(enumerate(frames), leave=False):
        # frame - это кадр
        # idx - это номер кадра
        mean = float(0)
        hist_mean = float(0)
        frame_features = {}
        if mask is None:
            height, width, _ = frame.shape
            mask = np.zeros((height, width), np.uint8)
            mask[(2 * height // 7):(5 * height // 7), (2 * width // 7):(5 * width // 7)] = 255

            mask_1q = frame[0 : height // 3, 0 : width // 3]
            mask_2q = frame[0 : height // 3, width // 3 : 2 * width // 3]
            mask_3q = frame[0 : height // 3, 2 * width // 3 : width]
            mask_4q = frame[height // 3 : 2 * height // 3, 0 : width // 3]
            mask_5q = frame[height // 3 : 2 * height // 3, width // 3 : 2 * width // 3]
            mask_6q = frame[height // 3 : 2 * height // 3, 2 * width // 3 : width]
            mask_7q = frame[2 * height // 3 : height, 0 : width // 3]
            mask_8q = frame[2 * height // 3 : height, width // 3 : 2 * width // 3]
            mask_9q = frame[2 * height // 3 : height, 2 * width // 3 : width]


        ### START CODE HERE ###
        frame_hist = hist(frame, mask, hist_size)

        frame_1q_hist = hist(mask_1q, None, hist_size)
        frame_2q_hist = hist(mask_2q, None, hist_size)
        frame_3q_hist = hist(mask_3q, None, hist_size)
        frame_4q_hist = hist(mask_4q, None, hist_size)

        frame_5q_hist = hist(mask_5q, None, hist_size)
        frame_6q_hist = hist(mask_6q, None, hist_size)
        frame_7q_hist = hist(mask_7q, None, hist_size)
        frame_8q_hist = hist(mask_8q, None, hist_size)
        frame_9q_hist = hist(mask_9q, None, hist_size)

        if prev_frame is not None:
            metric_value = pixel_metric(frame, prev_frame)
            hist_metric_value = hist_metric(frame_hist, prev_frame_hist)
            metric_values.append(metric_value)
            hist_metric_values.append(hist_metric_value)
            frame_features['metric_value'] = metric_value

            hist_metric_value_1q = hist_metric(frame_1q_hist, prev_frame_1q_hist)
            hist_metric_value_2q = hist_metric(frame_1q_hist, prev_frame_2q_hist)
            hist_metric_value_3q = hist_metric(frame_1q_hist, prev_frame_3q_hist)
            hist_metric_value_4q = hist_metric(frame_4q_hist, prev_frame_4q_hist)

            hist_metric_value_5q = hist_metric(frame_5q_hist, prev_frame_5q_hist)
            hist_metric_value_6q = hist_metric(frame_6q_hist, prev_frame_6q_hist)
            hist_metric_value_7q = hist_metric(frame_7q_hist, prev_frame_7q_hist)
            hist_metric_value_8q = hist_metric(frame_8q_hist, prev_frame_8q_hist)
            hist_metric_value_9q = hist_metric(frame_9q_hist, prev_frame_9q_hist)

            hist_metric_values_1q.append(hist_metric_value_1q)
            hist_metric_values_2q.append(hist_metric_value_2q)
            hist_metric_values_3q.append(hist_metric_value_3q)
            hist_metric_values_4q.append(hist_metric_value_4q)

            hist_metric_values_5q.append(hist_metric_value_5q)
            hist_metric_values_6q.append(hist_metric_value_6q)
            hist_metric_values_7q.append(hist_metric_value_7q)
            hist_metric_values_8q.append(hist_metric_value_8q)
            hist_metric_values_9q.append(hist_metric_value_9q)

            frame_features['hist_metric_value'] =  hist_metric_value
            frame_features['hist_metric_value_1q'] =  hist_metric_value_1q
            frame_features['hist_metric_value_2q'] =  hist_metric_value_2q
            frame_features['hist_metric_value_3q'] =  hist_metric_value_3q
            frame_features['hist_metric_value_4q'] =  hist_metric_value_4q

            frame_features['hist_metric_value_5q'] =  hist_metric_value_5q
            frame_features['hist_metric_value_6q'] =  hist_metric_value_6q
            frame_features['hist_metric_value_7q'] =  hist_metric_value_7q
            frame_features['hist_metric_value_8q'] =  hist_metric_value_8q
            frame_features['hist_metric_value_9q'] =  hist_metric_value_9q

            if idx >= frames_seq_len - 1:
                mean = float(np.nanmean(metric_values[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + metric_values[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean = float(np.nanmean(hist_metric_values[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values[idx - ((frames_seq_len + 1) // 2) + 2:]))
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['mean'] = mean
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean'] = hist_mean

                hist_mean_1q = float(np.nanmean(hist_metric_values_1q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_1q[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean_2q = float(np.nanmean(hist_metric_values_2q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_2q[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean_3q = float(np.nanmean(hist_metric_values_3q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_3q[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean_4q = float(np.nanmean(hist_metric_values_4q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_4q[idx - ((frames_seq_len + 1) // 2) + 2:]))

                hist_mean_5q = float(np.nanmean(hist_metric_values_5q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_5q[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean_6q = float(np.nanmean(hist_metric_values_6q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_6q[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean_7q = float(np.nanmean(hist_metric_values_7q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_7q[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean_8q = float(np.nanmean(hist_metric_values_8q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_8q[idx - ((frames_seq_len + 1) // 2) + 2:]))
                hist_mean_9q = float(np.nanmean(hist_metric_values_9q[-frames_seq_len: idx - ((frames_seq_len + 1) // 2) + 1] + hist_metric_values_9q[idx - ((frames_seq_len + 1) // 2) + 2:]))

                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_1q'] = hist_mean_1q
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_2q'] = hist_mean_2q
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_3q'] = hist_mean_3q
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_4q'] = hist_mean_4q

                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_5q'] = hist_mean_5q
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_6q'] = hist_mean_6q
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_7q'] = hist_mean_7q
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_8q'] = hist_mean_8q
                metrics[idx - ((frames_seq_len + 1) // 2) + 1]['hist_mean_9q'] = hist_mean_9q

            elif idx == frames_seq_len - 2:
                for i in range(0, (frames_seq_len) // 2):
                    mean = float(np.nanmean(metric_values[0: i] + metric_values[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    metrics[i]['mean'] = mean
                    hist_mean = float(np.nanmean(hist_metric_values[0: i] + hist_metric_values[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    metrics[i]['hist_mean'] = hist_mean

                    hist_mean_1q = float(np.nanmean(hist_metric_values_1q[0: i] + hist_metric_values_1q[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    hist_mean_2q = float(np.nanmean(hist_metric_values_2q[0: i] + hist_metric_values_2q[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    hist_mean_3q = float(np.nanmean(hist_metric_values_3q[0: i] + hist_metric_values_3q[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    hist_mean_4q = float(np.nanmean(hist_metric_values_4q[0: i] + hist_metric_values_4q[i + 1 : (frames_seq_len)// 2 + i + 1]))

                    hist_mean_5q = float(np.nanmean(hist_metric_values_5q[0: i] + hist_metric_values_5q[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    hist_mean_6q = float(np.nanmean(hist_metric_values_6q[0: i] + hist_metric_values_6q[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    hist_mean_7q = float(np.nanmean(hist_metric_values_7q[0: i] + hist_metric_values_7q[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    hist_mean_8q = float(np.nanmean(hist_metric_values_8q[0: i] + hist_metric_values_8q[i + 1 : (frames_seq_len)// 2 + i + 1]))
                    hist_mean_9q = float(np.nanmean(hist_metric_values_9q[0: i] + hist_metric_values_9q[i + 1 : (frames_seq_len)// 2 + i + 1]))

                    metrics[i]['hist_mean_1q'] = hist_mean_1q
                    metrics[i]['hist_mean_2q'] = hist_mean_2q
                    metrics[i]['hist_mean_3q'] = hist_mean_3q
                    metrics[i]['hist_mean_4q'] = hist_mean_4q
                    metrics[i]['hist_mean_5q'] = hist_mean_5q
                    metrics[i]['hist_mean_6q'] = hist_mean_6q
                    metrics[i]['hist_mean_7q'] = hist_mean_7q
                    metrics[i]['hist_mean_8q'] = hist_mean_8q
                    metrics[i]['hist_mean_9q'] = hist_mean_9q

        else:
            frame_features['metric_value'] = 0
            metric_values.append(0)
            frame_features['hist_metric_value'] = 0
            hist_metric_values.append(0)

            frame_features['hist_metric_value_1q'] =  0
            frame_features['hist_metric_value_2q'] =  0
            frame_features['hist_metric_value_3q'] =  0
            frame_features['hist_metric_value_4q'] =  0

            frame_features['hist_metric_value_5q'] =  0
            frame_features['hist_metric_value_6q'] =  0
            frame_features['hist_metric_value_7q'] =  0
            frame_features['hist_metric_value_8q'] =  0
            frame_features['hist_metric_value_9q'] =  0

            hist_metric_values_1q.append(0)
            hist_metric_values_2q.append(0)
            hist_metric_values_3q.append(0)
            hist_metric_values_4q.append(0)

            hist_metric_values_5q.append(0)
            hist_metric_values_6q.append(0)
            hist_metric_values_7q.append(0)
            hist_metric_values_8q.append(0)
            hist_metric_values_9q.append(0)

        prev_frame = frame
        prev_frame_hist = frame_hist
        prev_frame_1q_hist = frame_1q_hist
        prev_frame_2q_hist = frame_2q_hist
        prev_frame_3q_hist = frame_3q_hist
        prev_frame_4q_hist = frame_4q_hist

        prev_frame_5q_hist = frame_5q_hist
        prev_frame_6q_hist = frame_6q_hist
        prev_frame_7q_hist = frame_7q_hist
        prev_frame_8q_hist = frame_8q_hist
        prev_frame_9q_hist = frame_9q_hist
        metrics.append(frame_features)
        ###  END CODE HERE  ###
    for i in range(len(metric_values) - (frames_seq_len) // 2, len(metric_values)):
        mean = float(np.nanmean(metric_values[i - (frames_seq_len) // 2 : i] + metric_values[i + 1 : len(metric_values)]))
        metrics[i]['mean'] = mean
        hist_mean = float(np.nanmean(hist_metric_values[i - (frames_seq_len) // 2 : i] + hist_metric_values[i + 1 : len(metric_values)]))
        metrics[i]['hist_mean'] = hist_mean

        hist_mean_1q = float(np.nanmean(hist_metric_values_1q[i - (frames_seq_len) // 2 : i] + hist_metric_values_1q[i + 1 : len(metric_values)]))
        hist_mean_2q = float(np.nanmean(hist_metric_values_2q[i - (frames_seq_len) // 2 : i] + hist_metric_values_2q[i + 1 : len(metric_values)]))
        hist_mean_3q = float(np.nanmean(hist_metric_values_3q[i - (frames_seq_len) // 2 : i] + hist_metric_values_3q[i + 1 : len(metric_values)]))
        hist_mean_4q = float(np.nanmean(hist_metric_values_4q[i - (frames_seq_len) // 2 : i] + hist_metric_values_4q[i + 1 : len(metric_values)]))

        hist_mean_5q = float(np.nanmean(hist_metric_values_5q[i - (frames_seq_len) // 2 : i] + hist_metric_values_5q[i + 1 : len(metric_values)]))
        hist_mean_6q = float(np.nanmean(hist_metric_values_6q[i - (frames_seq_len) // 2 : i] + hist_metric_values_6q[i + 1 : len(metric_values)]))
        hist_mean_7q = float(np.nanmean(hist_metric_values_7q[i - (frames_seq_len) // 2 : i] + hist_metric_values_7q[i + 1 : len(metric_values)]))
        hist_mean_8q = float(np.nanmean(hist_metric_values_8q[i - (frames_seq_len) // 2 : i] + hist_metric_values_8q[i + 1 : len(metric_values)]))
        hist_mean_9q = float(np.nanmean(hist_metric_values_9q[i - (frames_seq_len) // 2 : i] + hist_metric_values_9q[i + 1 : len(metric_values)]))

        metrics[i]['hist_mean_1q'] = hist_mean_1q
        metrics[i]['hist_mean_2q'] = hist_mean_2q
        metrics[i]['hist_mean_3q'] = hist_mean_3q
        metrics[i]['hist_mean_4q'] = hist_mean_4q

        metrics[i]['hist_mean_5q'] = hist_mean_5q
        metrics[i]['hist_mean_6q'] = hist_mean_6q
        metrics[i]['hist_mean_7q'] = hist_mean_7q
        metrics[i]['hist_mean_8q'] = hist_mean_8q
        metrics[i]['hist_mean_9q'] = hist_mean_9q

    return metrics