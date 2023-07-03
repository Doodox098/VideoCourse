# Задание Neural Deinterlacing

### Использовавшееся окружение для ноутбука:
    ```
    conda create -n ENV_NAME matplotlib scikit-image scikit-learn tqdm pillow jupyter pytorch cudatoolkit=10.1 cudnn -c pytorch
    conda activate ENV_NAME
    ipython kernel install --name KERNEL_NAME --user
    ```

## Как обучить модель в ноутбуке
- Достаточно прогнать все ячейки ноутбука, вследствие чего будут инициализированы датасет, даталоадер, сама модель, показан пример из датасета и обучена модель с указанными размером батча `BATCH_SIZE` и числом эпох `EPOCHS`. Затем выведется результат работы модели и посчитанные метрики.
