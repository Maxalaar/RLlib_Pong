import tensorflow as tf

if __name__ == '__main__':
    # Définition d'une couche de convolution
    conv_layer = tf.keras.layers.Conv2D(
        filters=32,  # Nombre de filtres (ou noyaux) à utiliser
        kernel_size=(3, 3),  # Taille du noyau de convolution
        strides=(1, 1),  # Pas du déplacement du noyau
        # padding='valid',  # Type de padding ('valid' ou 'same')
        # activation='relu',  # Fonction d'activation (ReLU dans ce cas)
        input_shape=(84, 84, 4)  # Taille de l'entrée (84x84 pixels avec 4 canaux)
    )

    # Exemple d'utilisation de cette couche sur des données d'entrée
    input_data = tf.keras.Input(shape=(84, 84, 4))  # Création d'une couche d'entrée
    output_data = conv_layer(input_data)  # Application de la couche de convolution à l'entrée

    # Création d'un modèle Keras
    model = tf.keras.Model(inputs=input_data, outputs=output_data)

    # Affichage de la structure du modèle
    model.summary()
