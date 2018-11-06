def _ensemble(list_of_model_dir, model, val_x, val_y, batch_size = 100):
    _, H, W, _ = val_x.shape
    n_model = len(list_of_model_dir)
    total_pred = np.zeros_like(val_y) # (n_examples, n_classes)
    
    tf.reset_default_graph()

    X = tf.placeholder(tf.float32, [None, H, W, 1])
    y = tf.placeholder(tf.int64, [None, n_classes])
    
    _, y_out = model(X,y,is_training = False)
    
    with tf.Session(config = config) as sess:
        for checkpoint_path in list_of_model_dir:
            curr_pred = np.zeros_like(val_y)
            sess.run(tf.global_variables_initializer())
            saver = tf.train.Saver()
            saver.restore(sess, checkpoint_path)
            for step in range(int(val_x.shape[0] / batch_size)):
                offset = (step*batch_size) % (val_x.shape[0] - batch_size)
                batch_x = val_x[offset: offset+batch_size, :]
                batch_y = val_y[offset: offset+batch_size]
                feed_dict_val = {X: batch_x, y: batch_y}
                pred_y = sess.run(y_out,feed_dict=feed_dict_val)
                curr_pred[offset: offset+batch_size, :] = pred_y
            total_pred += curr_pred
        y_pred = tf.nn.softmax(total_pred)
        vote = np.argmax(y_pred, axis = -1)
        accuracy = np.mean(np.equal(vote, np.argmax(val_y, axis = -1)))
        return y_pred, vote
        print("Final validation accuracy with %d models: %.3f"%(len(list_of_model_dir),accuracy))