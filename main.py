if __name__ == '__main__':
    import load
    from dp_refined import Network

    train_samples, train_labels = load._train_samples, load._train_labels
    test_samples, test_labels = load._test_samples, load._test_labels

    print('Training set', train_samples.shape, train_labels.shape)
    print('    Test set', test_samples.shape, test_labels.shape)

    # image_size = load.image_size
    # num_labels = load.num_labels
    # num_channels = load.num_channels
    image_size = load.image_size
    image_size1= load.image_size1
    num_labels = load.num_labels
    num_channels = load.num_channels

    # def train_data_iterator(samples, labels, iteration_steps, chunkSize):
    #     """
    #     Iterator/Generator: get a batch of data
    #     这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
    #     用于 for loop， just like range() function
    #     """
    #     if len(samples) != len(labels):
    #         raise Exception('Length of samples and labels must equal')
    #     stepStart = 0  # initial step
    #     i = 0
    #     while i < iteration_steps:
    #         stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
    #         yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
    #         i += 1
    def train_data_iterator(samples, labels, iteration_steps, chunkSize):
        """
        Iterator/Generator: get a batch of data
        这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
        用于 for loop， just like range() function
        """
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        # ----------------------------------- CHANGED HERE -----------------------------------
        # reshuffle before each epoch to eliminate cyclic behavior
        from random import randint
        reshuffle = True
        stepStart = 0  # initial step
        lastStart = 0  # last step
        i = 0
        while i < iteration_steps:
            lastStart = stepStart
            stepStart = (i * chunkSize) % (labels.shape[0] - chunkSize)
            if reshuffle and stepStart < lastStart:
                for n in range(0,len(samples)):
                    tmp = randint(n,len(samples)-1)
                    tmp_image = samples[n]
                    samples[n] = samples[tmp]
                    samples[tmp] = tmp_image
                    tmp_label = labels[n]
                    labels[n] = labels[tmp]
                    labels[tmp] = tmp_label
            yield i, samples[stepStart:stepStart + chunkSize], labels[stepStart:stepStart + chunkSize]
            i += 1


    def test_data_iterator(samples, labels, chunkSize):
        """
        Iterator/Generator: get a batch of data
        这个函数是一个迭代器/生成器，用于每一次只得到 chunkSize 这么多的数据
        用于 for loop， just like range() function
        """
        if len(samples) != len(labels):
            raise Exception('Length of samples and labels must equal')
        stepStart = 0  # initial step
        i = 0
        while stepStart < len(samples):
            stepEnd = stepStart + chunkSize
            if stepEnd < len(samples):
                yield i, samples[stepStart:stepEnd], labels[stepStart:stepEnd]
                i += 1
            stepStart = stepEnd

    net = Network(
        train_batch_size=128, test_batch_size=120, pooling_scale=2,
        dropout_rate=0.9,
        base_learning_rate=0.0000005, decay_rate=0.41)
    net.define_inputs(
        train_samples_shape=(128, image_size, image_size1, num_channels),
        train_labels_shape=(128, num_labels),
        test_samples_shape=(120, image_size, image_size1, num_channels),
    )
    #
    net.add_conv(patch_size=3, in_depth=num_channels, out_depth=32, activation='relu', pooling=False, name='conv1')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv2')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=False, name='conv3')
    net.add_conv(patch_size=3, in_depth=32, out_depth=32, activation='relu', pooling=True, name='conv4')

    # 4 = 两次 pooling, 每一次缩小为 1/2
    # 32 = conv4 out_depth
    net.add_fc(in_num_nodes=(image_size // 4) * (image_size1 // 4) * 32, out_num_nodes=128, activation='relu',
               name='fc1')
    net.add_fc(in_num_nodes=128, out_num_nodes=6, activation=None, name='fc2')

    net.define_model()
    #net.run(train_samples, train_labels, test_samples, test_labels, train_data_iterator=train_data_iterator,
    #         iteration_steps=3000, test_data_iterator=test_data_iterator)
    net.train(train_samples, train_labels, data_iterator=train_data_iterator, iteration_steps=5000)
    net.test(test_samples, test_labels, data_iterator=test_data_iterator)

else:
    raise Exception('main.py: Should Not Be Imported!!! Must Run by "python main.py"')
