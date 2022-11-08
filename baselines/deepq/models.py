import tensorflow as tf


def build_q_func(network, hiddens=(256,), dueling=True, layer_norm=False, **network_kwargs):
    if isinstance(network, str):
        from baselines.common.models import get_network_builder
        network = get_network_builder(network)(**network_kwargs)

    def q_func_builder(input_placeholder, num_actions, scope, reuse=False):
        with tf.compat.v1.variable_scope(scope, reuse=reuse):
            latent = network(input_placeholder)
            if isinstance(latent, tuple):
                if latent[1] is not None:
                    raise NotImplementedError("DQN is not compatible with recurrent policies yet")
                latent = latent[0]

            latent = tf.keras.layers.Flatten()(latent)

            with tf.compat.v1.variable_scope("action_value"):
                action_out = latent
                for hidden in hiddens:
                    action_out = tf.compat.v1.layers.dense(action_out, hidden, None)
                    if layer_norm:
                        action_out = tf.compat.v1.layers.layer_norm(action_out, center=True, scale=True)
                    action_out = tf.nn.relu(action_out)
                action_scores = tf.compat.v1.layers.dense(action_out, num_actions, None)

            if dueling:
                with tf.compat.v1.variable_scope("state_value"):
                    state_out = latent
                    for hidden in hiddens:
                        state_out = tf.compat.v1.layers.dense(state_out, hidden, None)
                        if layer_norm:
                            state_out = tf.compat.v1.layers.layer_norm(state_out, center=True, scale=True)
                        state_out = tf.nn.relu(state_out)
                    state_score = tf.compat.v1.layers.dense(state_out, 1, None)
                action_scores_mean = tf.reduce_mean(input_tensor=action_scores, axis=1)
                action_scores_centered = action_scores - tf.expand_dims(action_scores_mean, 1)
                q_out = state_score + action_scores_centered
            else:
                q_out = action_scores
            return q_out

    return q_func_builder
