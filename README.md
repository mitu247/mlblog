# Guidance of a Diffusion Model with a Worse Version of Itself

**Author:** Sushmita Paul  
**Date:** 3/01/2025

Among the many interesting papers presented at NeurIPS 2024, one is **"Guiding a Diffusion Model with a Bad Version of Itself"** by NVIDIA. It deals with the intrinsically complex problem of creating generative models that emit high-quality outputs without compromising on variation or alignment with user-specified conditions. This blog goes over the problem, the proposed solution, and the implications for generative modeling.

---

## The Problem

Diffusion models form a powerful class of generative models synthesizing high-quality images by reversing a stochastic corruption process. Commonly, models are forced to make trade-offs along at least three axes of interest:

1. **Image Quality:** How realistic is the output?  
2. **Variation:** How diverse are the generated results?  
3. **Conditioning Alignment:** How well do the outputs adhere to user-specified prompts or labels?  

While the widely used **CFG** improves image quality and prompt alignment, this occurs at the expense of lowered variation. Unfortunately, these entangled effects seriously limit the practical applicability of CFG, especially in applications requiring both variety and fidelity. The authors thus believe that this behavior is rooted in differences in tasks between conditional and unconditional denoising networks.

---

## Diffusion

Diffusion modeling creates samples from a data distribution $p_{\text{data}}(x)$ by iteratively reversing the noise corruption process. To achieve this, one simulates the solution to a stochastic differential equation (SDE) or its deterministic variant—an ordinary differential equation (ODE).

### Forward Process

In the forward process, one iteratively adds noise to a data sample $x_0$, such that a sequence of increasingly noisy samples $x_t$ are created:

$$
q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\alpha_t} x_0, (1 - \alpha_t) \mathbf{I}),
$$

where $\alpha_t \in (0, 1)$ controls the noise schedule.

### Reverse Process

The reverse process removes noise step by step to recover $x_0$. This is formalized as:

$$
dx_t = -\nabla_x \log p(x_t) dt + \sqrt{2} dw_t,
$$

where $\nabla_x \log p(x_t)$ is the score function, and $dw_t$ represents Wiener noise.

In practice, the reverse process is parameterized by a neural network $D_\theta(x_t, t)$, which is trained to predict either the noise added or the denoised sample at each step:

$$L(\theta) = E_{x_0, t, \epsilon} [ \|D_\theta(x_t, t) - \epsilon\|^2 ]$$

where $\epsilon$ is the added noise.

---

## Key Contribution: Autoguidance

The paper introduces the approach of **Autoguidance**, which decouples improvement in image quality from variation control. Instead of relying on an unconditional model for guidance, this method uses a weaker version of the main model—a smaller or less-trained variant—as the guiding agent.

### Why Autoguidance Works

1. **Error Amplification:**  
   The weaker guiding model $D_0$ makes the same mistakes as the main model $D_1$ but does so more strongly. This acts as a directional signal to correct the output of the main model.

2. **Score-Based Guidance:**  
   Modifying the score function during guidance, the guidance can be done as:

   $$\nabla_x \log p_w(x|c;\sigma) = \nabla_x \log p_1(x|c;\sigma) + (w - 1) \nabla_x \log \frac{p_1(x|c;\sigma)}{p_0(x|c;\sigma)}$$

   where $w$ is the guidance weight, $p_1$ is the conditional density from the main model, and $p_0$ is the guiding model's density. This formula modifies the sampling trajectory, pulling outputs to be closer to the desired high-probability regions.

3. **Compatibility of Errors:**  
   For example, the degradations of the guiding model, such as reduced capacity or training time, align with the limitations of the main model and amplify shared deficiencies in low-probability regions.

---

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
