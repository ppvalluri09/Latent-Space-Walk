# Latent-Space-Walk

### Controlled Generation output

![alt text](https://raw.githubusercontent.com/ppvalluri09/Latent-Space-Walk/main/static/controlled_output.png)

<div align="center"><i>Face transition to Male</i></div>

### Abstract

Generative Adversarial Networks introduced by Ian Goodfellow in the year 2014 lead to a major breakthrough in the field of Generative Models with Deep Learning. Until then the VAEs were the king, and then GANs stepped in.

The great part about GANs is we no longer have to monitor if our models learns the distribution of the training set, inculcating loss term in terms of KL-Divergence for it. 

The idea of GAN is simple, let's we have a person <b>G</b> who wants to counterfeit currency, and a cop <b>D</b> who tries to catch the fake currency. G makes more and more counterfeit currency, while D tries to catch it. As D gets better at identifying fake currency from the real one, G tries to get better by fooling D that the counterfeit money is real. Hence it's a MinMax game, where G tries to minimize the bridge between fake and real money, whereas D tries to maximize. Eventually both get better at it.

We extend the same concept to GANs, we have a Generator <b>G</b> which tries to produce fake images and a Discriminator <b>D</b> which tries to tell the difference between fake and real images. Hence as D gets better and better, G gets better at fooling B. Obviously D is going to win, since it's always easier to tell if an image is real or fake than to generate it.

Vanilla GANs are awesome, but drawbacks?

1. Random Generation
2. Mode Collapse
3. Uncontrollable Generation

This project is all about addressing the third drawback. 

The dataset we used is the famous Celeba dataset, which is a collection of faces of famous celebrities. We go ahead and train our GAN with this dataset, as we do in a vanilla GAN. Training Configuration:-

```python3
training_config = {
	"EPOCHS": 50,
	"batch_size": 64,
	"lr": 0.001,
	"beta1": 0.5,
	"beta2": 0.99,
	"image_size": "64x64",
	"image_channels": 3
}
```

Now we use a technique called <b>Classifier Gradients</b> to obtain the noise vector we want. The noise vector is generated at random, which leads to uncontrallable nature of the GAN. But what if we had a pretrained classifier to classify the features of the human face, so that we can use that classifier to update the noise vector in the direction of the feature we want. So the celeba dataset comes with csv file containing the relevant features of the face for every image. SO we train a classifier and save it. Classifier training config:-

```python3
training_config = {
	"EPOCHS": 3,
	"batch_size": 64,
	"lr": 0.001,
	"beta1": 0.5,
	"beta2": 0.99,
	"image_size": 64,
	"image_channles": 3
}
```

Now we freeze the Classifier parameters, generate a random noise vector, pass each batch of images to the classifier, calculate the loss for a desired features, backprop and update the noise vector with a custom update rule.

### Checklist

  - [x] Built a GAN to generate human like faces
  - [x] Trained a Classifier to predict the features of a face
  - [x] Used Classifier Gradients to update noise vector to output desired feature
  - [ ] Disentanglement

### Acknowledgements

This project is just the implementation of the <a href="https://arxiv.org/pdf/1708.00598.pdf">Controllable Generative Adversarial Networks</a>

This is another paper that is a pre-requisite for to be able to comprehend CGANs <a href="https://arxiv.org/pdf/1406.2661.pdf">Generative Adversarial Networks</a>
