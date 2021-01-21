# Neural-Synth
A variational auto-encoder used to encode musical notes from the NSYNTH dataset with the goal of creating a continuous latent vector space with the encoder. Then manipulating that latent space and feeding it into the decoder to achieve notes of a different timber.

The model uses Binary Cross entropy for reconstruction loss and Kullback–Leibler divergence for latent loss. The greater the coefficient of the KL Divergence in the loss function, the more structured the latent space. However the model needs to get some initial training weights before KL divergence is implemented so that it doesnt minimise the KL divergence immedediately and get stuck. It should be a constant battle between reconstruction loss and Latent loss.

Unfortunately, I didn't have the hardware required to test on the full NSYNTH dataset so this designed for a small audio waveform.
