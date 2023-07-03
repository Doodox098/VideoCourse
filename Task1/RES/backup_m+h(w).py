def scene_change_detector(frames, threshold=None, with_vis=False):
    scene_changes = []
    vis = []
    metric_values = []
    hist_metric_values = []

    ### START CODE HERE ###
    # Ваши внешние переменные
    prev_frame = None
    prev_frame_hist = None
    hist_size = 16
    frames_seq_len = 7
    metrics_weight = 1.15
    hist_weight = 0.9
    hard_metrics_threshold = 11000

    def pixel_metric(frame, prev_frame):
        return np.mean((frame.astype(np.int32) - prev_frame) ** 2)
    def hist_metric(frame_hist,prev_frame_hist):
        return np.mean((frame_hist - prev_frame_hist) ** 2)
    def hist(frame):
        return cv2.calcHist(frame,[0], None, [hist_size], [0, 256])

    ###  END CODE HERE  ###

    for idx, frame in tqdm(enumerate(frames), leave=False):
        # frame - это кадр
        # idx - это номер кадра

        ### START CODE HERE ###
        pass
        # Основная часть вашего алгоритма
        frame_hist = hist(frame)
        if prev_frame is not None:
            metric_value = pixel_metric(frame, prev_frame)
            hist_metric_value = hist_metric(frame_hist, prev_frame_hist)
            metric_values.append(metric_value)
            hist_metric_values.append(hist_metric_value)

            if idx > frames_seq_len:
                mean = np.mean(metric_values[-frames_seq_len:])
                hist_mean = np.mean(hist_metric_values[-frames_seq_len:])
                if (metric_values[idx - ((frames_seq_len + 1) // 2) + 1] / mean * metrics_weight + hist_metric_values[idx - ((frames_seq_len + 1) // 2) + 1] / hist_mean * hist_weight) > 8.0:
                    scene_changes.append(idx - ((frames_seq_len + 1) // 2) + 1)
        else:
            metric_values.append(0)
            hist_metric_values.append(0)
        prev_frame = frame
        prev_frame_hist = frame_hist

        ###  END CODE HERE  ###

    return scene_changes, vis, metric_values