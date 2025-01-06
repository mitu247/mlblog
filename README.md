# Guiding a Diffusion Model with a Bad Version of Itself

**Author:** Sushmita Paul  
**Date:** 3/01/2025

So, you know about image generation, right? Well, this is the domain of **Generative Models** which can do magical things like creating images from just from your thoughts. Who wouldn't be happy if they could just imagine a beautiful landscape and see it right in front of their eyes?

<p align="center">
  <img src="AI_Image_Gen.webp" alt="AI Image Generation" width="500" height="300">
  <br>
  <em>Figure 1: Dream into Reality</em>
</p>

But, as with all things, there are challenges. One of the biggest challenges in generative modeling is to create models that can generate high-quality images with a lot of variety.

Buckle up fellow dreamers, because I am going to take you through a paper that addresses this very challenge. The paper is titled **"Guiding a Diffusion Model with a Bad Version of Itself"** by NVIDIA. Let's dive in!


## The Problem

Currently, the image generation community faces a dilemma: how to balance image **quality** and **variation**. At present, the most popular technique to guide image generation is **Classifier-Free Guidance (CFG)**. This technique uses an unconditional model to guide a conditional one, thereby enhancing prompt alignment and image quality but comes at the cost of reduced variation.Unfortunately, these entangled effects seriously limit the practical applicability of CFG, especially in applications requiring both variety and fidelity.

<p align="center">
  <img src="Improvement_Example.png" alt="Improved Image Generation Example" width="500" height="300">
  <br>
  <em>Figure 2: Example results for the Tree frog, Palace, Mushroom, Castle</em>
</p>

So, they focused on developing a new technique called **Autoguidance** which adresses problems regarding-
1. **Image Quality:** How realistic is the output?  
2. **Variation:** How diverse are the generated results?  
3. **Conditioning Alignment:** How well do the outputs adhere to user-specified prompts or labels?  

Good News!:tada: The researchers at NVIDIA discovered that by guiding the generation model with a smaller and less-trained version of the model itself, high-quality image generation without sacrificing variation is possible (See _Figure 2_). This breakthrough offers disentangled control over image quality and variation, making it a game-changer in image generation.

## Some Theoritical References

### Diffusion Modeling

It creates samples from a data distribution $p_{\text{data}}(x)$ by iteratively reversing the noise corruption process. To achieve this, one simulates the solution to a stochastic differential equation (SDE) or its deterministic variant—an ordinary differential equation (ODE).

### Forward Process (**NOISE-UP**)

In the forward process, one iteratively adds noise to a data sample $x_0$, such that a sequence of increasingly noisy samples $x_t$ are created:

$$
\begin{equation}
p(x; \sigma) = p_{\text{data}}(x) * N(x; 0, \sigma^2 I)
\end{equation}
$$

The equation  describes smoothing a data distribution by convolving it with a Gaussian, which for large $( \sigma )$ approximates white noise, allowing easy sampling from a normal distribution.

### Reverse Process (**NOISE-DOWN**)

The reverse process removes noise step by step to recover $x_0$. This is formalized as:

$$
\begin{equation}
dx_\sigma = -\sigma \nabla_{x_\sigma} \log p(x_\sigma ; \sigma) d\sigma
\end{equation}
$$ 

The equation describes a probability flow ODE that evolves a sample from high to low noise levels, maintaining the distribution $p(x_\sigma ; \sigma)$ and ultimately recovering the original data distribution $p_{\text{data}}(x_0)$ when $( \sigma = 0 )$.

The ODE is solved numerically by stepping along the trajectory defined by Eq. (1), requiring the evaluation of the score function $\nabla_x \log p(x; \sigma)$ for a given sample $x$ and noise level $\sigma$. This can be approximated using a neural network $D_\theta (x; \sigma)$ trained for denoising:

$$
\theta = \arg \min_\theta \mathbb{E}_{y \sim p_{\text{data}}, \sigma \sim p_{\text{train}}, n \sim N(0, \sigma^2 I)} \| D_\theta (y + n; \sigma) - y \|_2^2,
$$

where $p_{\text{train}}$ controls the noise level distribution during training with score function is estimated as:

$$
\nabla_x \log p(x; \sigma) \approx \frac{D_\theta (x; \sigma) - x}{\sigma^2}.
$$

Each data sample $x$ is associated with a label $c$. At generation time, we control the outcome by choosing $c$ and seeking a sample from $p(x|c; \sigma)$ with $\sigma = 0$, achieved by training $D_\theta (x; \sigma, c)$ with $c$ as an additional input.

## Why Autoguidance Works ??

1. **Error Amplification:**
   The weaker guiding model $D_0$ makes the same mistakes as the main model $D_1$ but does so more **strongly**. This acts as a directional signal to correct the output of the main model. Something like a **"bad cop"** guiding a **"good cop"** to make better decisions.

2. **Score-Based Guidance:**
   Modifying the score function during guidance, the guidance can be done as:

   $$\nabla_x \log p_w(x|c;\sigma) = \nabla_x \log p_1(x|c;\sigma) + (w-1)\nabla_x \log \frac{p_1(x|c;\sigma)}{p_0(x|c;\sigma)}$$

   where $w$ is the guidance weight, $p_1$ is the conditional density from the main model, and $p_0$ is the guiding model's density. This formula modifies the sampling trajectory, pulling outputs to be closer to the desired high-probability regions.

3. **Compatibility of Errors:**  
   For example, the degradations of the guiding model, such as <u>reduced capacity</u> or <u>training time</u>, align with the limitations of the main model and amplify shared deficiencies in low-probability regions.

## Results and Impact

The authors evaluate autoguidance on the ImageNet dataset, achieving state-of-the-art FID scores:

- **1.25** for ImageNet-512 (512×512 resolution)  
- **1.01** for ImageNet-64 (64×64 resolution)  

### Quantitative Results

Autoguidance surpasses the current state-of-the-art in both CFG and interval-based guidance techniques. The paper highlights:

- **47% FID improvement** on ImageNet-512 compared to CFG.  
- Preservation or even improvement in variation while increasing the quality of the images.

### Qualitative Results

Autoguidance gives more diverse and realistic outputs. Consider the following examples:

- In the "Palace" class, CFG simplifies compositions into canonical templates, whereas autoguidance preserves rich, atypical details.  
- In complex scenes, autoguidance focuses on enhancing individual elements without sacrificing overall diversity.

---

## Personal Insights and Future Directions

### My Interpretation

Autoguidance represents a paradigm shift in generative modeling. The method uses a "bad version" of the model itself in an elegant solution to a long-standing issue of balancing quality and diversity. Besides being practical, this approach has something unusually philosophically intriguing to it—leveraging imperfection as a means to refinement.

### Possible Applications

1. **Unconditional Generation:**  
   Autoguidance improves unconditional diffusion models, demonstrated by reducing FID from 11.67 to 3.86 in ImageNet experiments.

2. **Creative Industries:**  
   Artists and designers might welcome more diversity and fidelity in generative tools.

3. **Scientific Simulations:**  
   Applications like climate modeling or material design that require both accuracy and diversity could benefit from autoguidance-inspired techniques.

### Future Research

1. **Theoretical Guarantees:**  
   Formulate theoretical results on when exactly autoguidance improves over CFG.

2. **Optimized Guiding Models:**  
   Investigate architectural or training optimizations that better align $D_1$ and $D_0$.

3. **Cross-Domain Applications:**  
   Extend autoguidance principles into text, video, and 3D generative models.

---

## Conclusion

This work, "Guiding a Diffusion Model with a Bad Version of Itself," testifies to the ingenuity of researchers tackling core challenges in generative modeling. Autoguidance enables quality and variation control to be decoupled, making generative models more robust and varied. These approaches will undoubtedly form the foundation for future innovations in diffusion modeling.

---

## References

1. **NVIDIA.** "Guiding a Diffusion Model with a Bad Version of Itself." NeurIPS 2024. [Paper Link](https://openreview.net/forum?id=bg6fVPVs3s)
2. **AI Image Generation Picture** from [Medium](https://medium.com/@natiberk/the-state-of-ai-image-generation-03-24-e91f6d7ea6cf)
