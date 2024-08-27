# Generative Flow Networks for Probabilistic Surrogates: Addressing High-Dimensional Cost Challenges

**Honor's research project at Université de Montréal**

This project introduces an innovative framework designed to tackle the growing computational challenges faced by researchers working with complex, high-dimensional data across a variety of scientific fields. By integrating Gaussian Processes (GPs) with Generative Flow Networks (GFNs), we’ve developed a new approach to surrogate modelling that significantly enhances both efficiency and accuracy. Traditional methods for sampling complex oracles, such as those used in fields like cosmology, often become impractical due to the sheer computational costs involved. Our framework addresses this problem by using GPs as surrogate models to approximate these costly evaluations while keeping resource demands in check.

The unique addition of GFNs as the acquisition model is a key innovation in our approach. Unlike standard Bayesian Optimization techniques, which are primarily designed for function optimization, GFNs employ reinforcement learning to learn optimal sampling strategies. This enables the model to effectively explore the input space and identify the most informative samples, thereby improving the surrogate model's accuracy without excessive computational overhead. Moreover, our method incorporates a carefully designed reward function that balances exploration and exploitation, ensuring that each new sample significantly improves the surrogate model's performance.

Our research also explores the potential of further enhancing this methodology by tweaking the reward functions and incorporating advanced models such as Set Transformers to preprocess inputs, which could help the GFNs detect and leverage patterns in the data more effectively. We acknowledge that there is a high initial computational cost associated with training the GFN. However, this investment is offset by the long-term efficiencies gained in future evaluations, making it a more sustainable alternative for ongoing scientific investigations.

As we continue to develop and refine this framework, we’re excited about its potential to make a significant impact in fields that rely on large-scale data and simulations, from physics and climate science to finance and engineering. We believe this project has the potential to become a key tool in the scientific community, driving advancements and enabling new discoveries by making complex data analysis more accessible and efficient.

## Structure

- `./src/`
  - Jupyter notebooks demonstrating a high level use of the models
  - Python files used for the backend code
- `./src/models/`
  - Checkpoints of some models
- `./docs-src/`
  - Contains the source files of the project's website
  - That website is compiled into `./docs/` and served from there
- `./report/`
  - Contains all of the files relevant to the report
