def scene_change_detector(frames, threshold=None, with_vis=False):
    scene_changes = []
    vis = []
    metric_values = []

    ### START CODE HERE ###
    # Ваши внешние переменные
    prev_frame = None
    frames_seq_len = 7
    metrics_threshold = 3.2
    hard_metrics_threshold = 11000

    def pixel_metric(frame, prev_frame):
        return np.mean((frame.astype(np.int32) - prev_frame) ** 2)

    ###  END CODE HERE  ###

    for idx, frame in tqdm(enumerate(frames), leave=False):
        # frame - это кадр
        # idx - это номер кадра

        ### START CODE HERE ###
        pass
        # Основная часть вашего алгоритма
        if prev_frame is not None:
            metric_value = pixel_metric(frame, prev_frame)
            metric_values.append(metric_value)

            if idx > frames_seq_len:
                mean = np.mean(metric_values[-frames_seq_len:])
                if metric_values[idx - ((frames_seq_len + 1) // 2) + 1] / mean > metrics_threshold or metric_values[idx - ((frames_seq_len + 1) // 2) + 1] > hard_metrics_threshold:
                    scene_changes.append(idx - ((frames_seq_len + 1) // 2) + 1)
        else:
            metric_values.append(0)
        prev_frame = frame

        ###  END CODE HERE  ###

    return scene_changes, vis, metric_values