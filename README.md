# NeurIPS 2018 Notes

[YJ Choe](https://yjchoe.github.io/)

Below are my mostly unedited notes from NeurIPS 2018. 
Also check out [**my top 10 picks from the conference**](https://yjchoe.github.io/blog/distillation/2018/12/31/my-top-10-picks-from-neurips-2018.html).

---

# Conference Day 1 (2018.12.04 Tuesday)

## Test of Time Award: The Trade-offs of Large Scale Learning

*Tue Dec 4th 09:20 -- 09:40 AM*

(cf. last year's award talk was quite something: [https://youtu.be/Qi1Yry33TQE](https://youtu.be/Qi1Yry33TQE))

Bottou & Bousquet*, NIPS 2007

- This was a paper that advocated stochastic (i.e. online, as opposed to batch) training for large-scale training of kernel machines.
- The paper formalized the three-way decomposition the expected loss: approximation, estimation, and optimization errors.
- Message: **Spend less time and see more data!**
- Future directions
    - Variance reduction: stochastic gradients are noisy.
    - Implicit regularization with SGD: well, stochastic gradients are noisy.
    - Robust algorithms: coping with the sensitivity of SGD to step size, for example.
- Thoughts
    - Scientific truth does not follow the fashion
        - Someone should continue to work on SGDs.
    - Experiments are crucial
        - Not those beating SOTA, but those that reveal insights: validate or invalidate claims.
    - Proper use of mathematics
        - A theorem is not all.
- Three future problems
    - The role of over-parametrization in learning and generalization.
    - The role of compositionality in ML.
        - More than just train→test. Multiple tasks, losses, ...
    - The role of causality in AI.
        - Humans are far better than machines in making causal claims & inference.

## On Neuronal Capacity

*Tue Dec 4th 10:05 -- 10:20 AM @ Room 220 CD*

- NN Capacity: the logarithm of the number of functions that in the hypothesis class.
    - Also represents the number of bits stored in a network.
- The capacity of neurons in networks can be bounded, and this work gives a (highly non-trivial) bound on the polynomial-threshold gates.
    - n^(d+1)/(d!)
    - [n = number of incoming neurons, d = degree of polynomials]
- Extension to networks: simple upper bound as the sum of neuronal capacities.
    - RNN: n^3
    - Layered FC Feed-forward: cubic polynomial in the size of the layers (\sum_{k=1}^{L-1} min(n_1, ..., n_k) n_kn_k+1)
- **Everything else equal, deep architectures have less capacity, but the functions they compute are more regular!**
    - 1-hidden-layer with m hidden units: mn^2
    - L-hidden-layer with m hidden units: mn^2 + mn^3
    - “Structural Regularization”: deep architectures are bounded to estimate “nicer” functions.

## Learning Overparameterized Neural Networks via Stochastic Gradient Descent on Structured Data

*Tue Dec 4th 10:20 -- 10:25 AM @ Room 220 CD*

- Overparametrized NNs: the optimization is highly non-convex, but still appears to give nice solutions.
- Mystery: The optimization magically figures out the structure of the data!
    - Wider networks seem easier to train.
    - Practical deep NNs fit random labels.
- Is there a simple theoretical explanation? Yes, for two-layer NNs on clustered data. (Come see the poster!)

## Size-Noise Tradeoffs in Generative Networks

*Tue Dec 4th 10:25 -- 10:30 AM @ Room 220 CD*

- Wasserstein error bounds on generative networks
    - Input dim < output dim: (width)^(depth)
    - =, < cases also analyzed in the paper.

## A Retrieve-and-Edit Framework for Predicting Structured Outputs

*Tue Dec 4th 10:30 -- 10:45 AM @ Room 220 E*

- Generate structured outputs (e.g. code) by first retrieving a similar-looking training example and then editing it.
- Related to structured prediction, memory, and/or nonparametric regression.
- Cf. Prototype-then-edit

---

## Tuesday Poster Session A (10:45 AM — 12:45 PM)

168 papers

- Noteworthy
    - **Towards Text Generation with Adversarially Learned Neural Outlines**
        - An off-the-shelf encoder embedding is pitted against a critic before going into a decoder network.
        - An adversarial generator is an MLP that learns to generate “fake sentence embeddings.”
    - **Recurrently Controlled Recurrent Networks**
        - Forget & output gates are replaced with controller RNN cells that (are supposed to) control memory (forgetting) and output prediction.
        - Competitive performance on classification tasks.
        - (Use as a pre-trained language model?) Would be computationally intensive.
    - **Dendritic cortical microcircuits approximate the backpropagation algorithm**
        - Bioplausible algorithms... NOT replacing but supporting backprop!
        - “Pyramidal neuron learning”
    - **DeepPINK: reproducible feature selection in deep neural networks**
        - A deep learning version of knockoff FDR control.
        - Can be combined with KnockoffGAN for a fully end-to-end feature selection.
    - **Latent Alignment and Variational Attention**
        - A probabilistic formulation of the attention mechanism.
    - **Middle-Out Decoding**
        - Start a sentence from the middle and go left-right-left-right-...
        - Could make sense with visual storytelling, among other things.
        - (Would BERT-style all-to-all decoding work?) Doubtful.
    - **Complex Gated Recurrent Neural Networks**
        - Complex-valued RNN cells can be made!
        - Application to human motion prediction.
        - Speech, text, ... all future work.
    - **Expanding Holographic Embeddings for Knowledge Completion**
        - Provides a middle-ground between the expressive RESCAL and the efficient holographic embeddings by a perturbing the head embedding.
    - **Content preserving text generation with attribute controls**
    - **Representer Point Selection for Explaining Deep Neural Networks**
        - “We propose to explain the predictions of a deep neural network, by pointing to the set of what we call representer points in the training set, for a given test point prediction. *Specifically, we show that we can decompose the pre-activation prediction of a neural network into a linear combination of activations of training points, with the weights corresponding to what we call representer values, which thus capture the importance of that training point on the learned parameters of the network.”*

- Other notable papers
    - On Dimensionality of Word Embedding
    - Glow: Generative Flow with Invertible 1x1 Convolutions
    - Variational Memory Encoder-Decoder
    - On the Global Convergence of Gradient Descent for Over-parameterized Models using Optimal Transport
    - DVAE#: Discrete Variational Autoencoders with Relaxed Boltzmann Priors
    - IntroVAE: Introspective Variational Autoencoders for Photographic Image Synthesis
    - Generalizing Point Embeddings using the Wasserstein Space of Elliptical Distributions
    - L4: Practical loss-based stepsize adaptation for deep learning
    - Relational recurrent neural networks
    - Recurrent Relational Networks
    - GLoMo: Unsupervised Learning of Transferable Relational Graphs
    - Diffusion Maps for Textual Network Embedding
    - Modern Neural Networks Generalize on Small Data Sets
    - Spectral Filtering for General Linear Dynamical Systems
    - Reversible Recurrent Neural Networks

---

## What Bodies Think About: Bioelectric Computation Outside the Nervous System, Primitive Cognition, and Synthetic Morphology

*Tue Dec 4th 02:15 -- 03:05 PM @ Rooms 220 CDE*

Invited Talk.

- Artificial neural networks imitate brains, but are brains the best model? How does the anatomical structure of biological organisms evolve?
    - Seeking biological evidences of algorithms *beyond manipulation of molecules and cells.*
    - What algorithms explain the *control* of large-scale forms? (Bioelectric computations.)
- Future: **Could a highly robust (non-brittle) ML roadmap be based on non-neural architectures?**
- (Question) Consciousness?
    - Not going to argue (lots of people think about this stuff), but there is an emergent field of Bayesian cognition.
    - Everything is within a continuum: no one concept (including consciousness) belongs solely to human cognition.
- (Question) What non-neural systems?
    - Robust adaptive systems. Biological systems (e.g. tissues) are very good at adapting to novel environments!
    - Biology has competency at every level. At each level, there are very novel and intelligent components with specific goals.

---

## Evolved Policy Gradients

*Tue Dec 4th 03:30 -- 03:35 PM @ Room 220 E*

- Evolution Strategies + Stochastic Gradient Descent
- Generalization beyond training task distribution (meta-learning task): ant following targets in different positions.

## Adapted Deep Embeddings: A Synthesis of Methods for k-Shot Inductive Transfer Learning

*Tue Dec 4th 03:35 -- 03:40 PM @ Room 220 E*

- Inductive Transfer Learning
    - Weight Transfer (#labels-per-class ≥ 100)
    - Deep Metric Learning (based on inter-class separation)
    - Few-Shot Learning (#labels-per-class ≤ 20): semi-supervised methods (prototypical nets), meta-learning, ...
- Adapted Deep Embeddings: Combine the three methods.
    - Best performance when all methods are effectively combined.
        - Weight transfer is the least effective.
        - Histogram loss is robust to the number of labels.
        - Adapted embeddings outperform.

## Bayesian Model-Agnostic Meta-Learning

*Tue Dec 4th 03:40 -- 03:45 PM @ Room 220 E*

- A Bayesian version of MAML
    - Uncertainty provides robustness to over-fitting
- Resolves the problems of two previous work (MAML + LLAMA)
    - Bayesian fast-adaptation (BFA): enabled by Stein variational gradient descent (SVGD)
    - Chaser loss: chaser particles computed using SVGD, and some particles are given by additional iterations, but made sure to not deviate from the “leader” too much via the chaser loss.

## Neural Ordinary Differential Equations

*Tue Dec 4th 03:50 — 04:05 PM @ Room 220 E*

One of the Best Papers!

- Intimate connection between neural networks and ordinary differential equations.
    - A modification of ResNet leads to an ODE operation!
    - **ODENet**: improvement over the modification coming from an efficient ODE solver.
- A drop-in replacement for ResNets
    - Achieves the same performance as ResNets
- Don’t have a fixed number of layers: the depth is automatically set by the ODE solver!
    - 2-4x depth from ResNet
    - Downside: we can’t control the time cost during training. (Can control in test time, though)
- Explicit error control based on the ODE solver’s tolerance.
- Can extend to continuous-time models.
    - Much improved performance on synthetic data than RNNs!
- Best application: density estimation.
    - Instantaneous change of variables: requires trace instead of determinant
    - Leads to: **continuous normalizing flows**
    - Allows continuous transformation of a Gaussian into the desired density (e.g. images)!

## Bias and Generalization in Deep Generative Models: An Empirical Study

*Tue Dec 4th 04:05 -- 04:10 PM @ Room 220 E*

- Examination of deep generative models (their samples)
- Numerosity: if given training samples that always contain two objects, do models generate two-object images only?
    - No. Mode is 2, but also generates 1/3/4-object images.
    - If given 2/4 objects, model generates 1~6 object images, but the mode is 3 (cf. psychology)
- Multiple features: colors, shapes, ...
    - Few number of features: model memorizes.
    - Many number of features seen: can generalizes. (Sharp threshold!)
- Similar behaviors, regardless of the model choice.

## Robustness of conditional GANs to noisy labels

*Tue Dec 4th 04:10 — 04:15 PM @ Room 220 E*

- Conditional GANs suffer from label noise.
- Robust Conditional GAN (rcGAN) is proposed.
    - Minimize divergence between noisy and clean distributions.
    - Neural network distance is used.

## BourGAN: Generative Networks with Metric Embeddings

*Tue Dec 4th 04:15 -- 04:20 PM @ Room 220 E*

- Mode collapse: model only covers a subset of the existing modes.
- Bourgain’s Theorem is the motivation.
- Use GMM & encourage the modes to be distant from each other.

## How Does Batch Normalization Help Optimization?

*Tue Dec 4th 04:25 -- 04:40 PM @ Room 220 E*

- A closer look at “Internal Covariance Shift”
    - Doesn’t seem to be the case upon examination, with or without batch norm.
    - “Noisy” BatchNorm: adding distributional instability has almost NO IMPACT on optimization or performance.
- Different view: gradients
    - Using BatchNorm yields no apparent decrease in terms of the gradient change after previous layer has been updated.
- A landscape view: how does the loss change across the gradient direction?
    - **BatchNorm does have a profound impact on this landscape!**
    - Loss fluctuates much less across the landscape.
- The effect of a *single* BatchNorm layer
    - Loss becomes more Lipschitz.
    - Gradient becomes more predictive.
    - Translates into similar worst-case improvements.

## Neural Architecture Search with Bayesian Optimisation and Optimal Transport

*Tue Dec 4th 04:45 -- 04:50 PM @ Room 517 CD*

- Define a kernel between neural networks.
- Apply Bayesian optimization → Neural Architecture Search!
- NASBOT: Bayesian optimization, optimal transport, and evolutionary algorithms.

## Supervised Unsupervised Learning

*Tue Dec 4th 04:55 -- 05:00 PM @ Room 517 CD*

- Define a meta-distribution over all problems in the universe (!)
- An evaluation metric for unsupervised methods and a threshold for single-linkage clustering.
- “Meta-clustering” is presented.

---

## Tuesday Poster Session B (05:00 PM — 07:00 PM)

170 papers

- Noteworthy
    - **To Trust Or Not To Trust A Classifier**
        - Compare my model’s prediction with a test-set nearest-neighbor classifier.
        - Do NOT trust it if its prediction differs too much from the NN classifier.
    - **Sparse Attentive Backtracking: Temporal Credit Assignment Through Reminding**
        - Only remember to backprop every K steps (or top-K?) to allow efficient credit assignment.
        - Claims competitive performance against Transformers on classficiation tasks.

- Other notable papers
    - **Generating Informative and Diverse Conversational Responses via Adversarial Information Maximization**
    - **e-SNLI: Natural Language Inference with Natural Language Explanations**
    - Step Size Matters in Deep Learning
    - Deep Generative Markov State Models
    - Generative modeling for protein structures
    - Layer-Wise Coordination between Encoder and Decoder for Neural Machine Translation
    - Dialog-to-Action: Conversational Question Answering Over a Large-Scale Knowledge Base
    - Neural-Symbolic VQA: Disentangling Reasoning from Vision and Language Understanding
    - Beauty-in-averageness and its contextual modulations: A Bayesian statistical account
    - Model-Agnostic Private Learning
    - Generative Probabilistic Novelty Detection with Adversarial Autoencoders
    - Neural Nearest Neighbors Networks
    - Answerer in Questioner's Mind: Information Theoretic Approach to Goal-Oriented Visual Dialog
    - Attacks Meet Interpretability: Attribute-steered Detection of Adversarial Samples
    - Distilled Wasserstein Learning for Word Embedding and Topic Modeling
    - Masking: A New Perspective of Noisy Supervision
    - Data-Efficient Hierarchical Reinforcement Learning

---

## TextWorld: A Learning Environment for Text-based Games (Demo)

*Tue Dec 4th 10:45 AM -- 07:30 PM @ Room 510 ABCD #D10*

An all-day demo by the original authors of TextWorld from Microsoft Research. Nothing too special, though, just trying out the TextWorld environment as a demo.

## Ruuh: A Deep Learning Based Conversational Social Agent (Demo)

*Tue Dec 4th 10:45 AM -- 07:30 PM @ Room 510 ABCD #D3*

“Ruuh” ([facebook.com/Ruuh](http://facebook.com/Ruuh)) is a conversational social agent developed by a team at Microsoft India. It claims to be an agent that do better than merely generating relevant responses (expressing happiness when user's favorite team wins, sharing a cute comment on the user's pet, detecting and responding to abusive language, sensitive topics and trolling behaviors, and so on). It has interacted with over 2 million real world users, generating over 150 million conversations.

---

# Conference Day 2 (2018.12.05)

## Reproducible, Reusable, and Robust Reinforcement Learning

*Wed Dec 5th 08:30 -- 09:20 AM @ Rooms 220 CDE*

Invited Talk.

- This talk provided an impressive reality-check on the current status of reinforcement learning research.
    - Among the 50 RL papers published at NeurIPS, ICML, and ICLR in 2018, all papers have experiments, most have hyperparameters provided, BUT only 20% reports *how* the hyperparameters are chosen and an appalling 5% reports significance testing results.
    - Worst part: nobody explains *what* the shades are! Standard error? 95% confidence interval? 68%?
    - Introducing the reproducibility checklist.
- Towards generalization in RL
    - Myth or Fact? RL is the only case of ML where it is acceptable to test on your training set.
        - There is an immense gap between performance in simulated environments (e.g. MuJoCo) and in the real world. There are several steps in the middle:
            - Separating the training and testing set is still a challenge in RL.
            - Even separating the randomj seed between training and testing (and, of course, using sufficiently many random trials) helps! Not all hope is lost. (Note: 5 instead of 1 seed would help within the simulator, but for the real-world you may need 100 seeds.)
            - Natural world is so much more complex, so one may want to incorporate photorealistic videos within the simulators (at the very least) as natural noise.
- Conclusions
    - Science is a **collective, not competitive,** process.
    - Check out the reproducibility checklist.
    - Check out the ICLR Reproducibility Challenge. (80% of the authors appreciate it and make the changes!)

---

## Text-Adaptive Generative Adversarial Networks: Manipulating Images with Natural Language

*Wed Dec 5th 09:55 -- 10:00 AM @ Room 220 E*

- GANs conditional on textual cues that, unlike previous work, does not alter the background or other image contents unrelated to the text.
- An additional text-adaptive discriminator network is used.
- Outperforms other relevant methods in terms of image quality and background preservation.

## Neighbourhood Consensus Networks

*Wed Dec 5th 10:00 -- 10:05 AM @ Room 220 E*

- Find pixel-level image correspondences across day-night, time, viewpoint, or other variations.
- Neighborhood consensus: number of spatially consistent matches are considered as correct.
    - All pairwise matches are initially considered → a neighborhood consensus network filters among the N^2 possible matches.
    - Only image-level supervision is required (a positive/negative label for image pairs).
- Impressive results across quite large variations (different buses pictured at different angles, ...)

## Visual Memory for Robust Path Following

*Wed Dec 5th 10:05 -- 10:20 AM @ Room 220 E*

- On navigation
    - Situated in an unknown environment, a mice initially explores, but it already employs a systematic mechanism that effectively exploits the system. Not to mention humans.
    - Only key moments need to be remembered: don’t need to reconstruct all the details.
    - Path planning + Path following.
- Robust Path Follower (RPF): a recurrent neural network “follows” along the checkpoints along a familiar path.
    - Localize only when necessary.
    - Achieves competitive performance by planning only based on the necessary cues.

## Recurrent Transformer Networks for Semantic Correspondence

*Wed Dec 5th 10:20 -- 10:25 AM @ Room 220 E*

- RTN: recurrently model the residuals using a Transformer.
- Use a Siamese-style “geometric matching” network to model semantic correspondence.
- Weakly-supervised training is used.

## Sequential Attend, Infer, Repeat: Generative Modelling of Moving Objects

*Wed Dec 5th 10:25 -- 10:30 AM @ Room 220 E*

- Generative modeling for unsupervised motion detection.
- Attend, Infer, Repeat (AIR): a VAE that decomposes an image into objects.
    - Objects are modeled using separate latent variables (what, where, presence)
- Sequential AIR (SQAIR): a dynamic variant.
    - One latent variable per object.
    - Knows identity (and tracks it).
    - Sampling from the model generates a video of objects moving across frames.
    - Reconstruction is possible because of hints from other time frames.

## Sanity Checks for Saliency Maps

*Wed Dec 5th 10:30 -- 10:35 AM @ Room 220 E*

- Examining the “evidence” behind the prediction using saliency maps is standard. But can we trust them?
- Hypothesis: if the prediction is “completely garbage” (using randomized weights), the explanations should change.
    - **But the explanations don’t change! A bunch of false positives.**
    - Most methods (GradCAM, Integraded Gradients, ...) fail.
    - “Classic confirmation bias”: just because it makes sense to humans, doesn’t mean it reflects the evidence for prediction.

---

## Wednesday Poster Session A (10:45 AM — 12:45 PM)

169 papers

- Noteworthy
    - **Blockwise Parallel Decoding for Deep Autoregressive Models**
        - A tractable & effective middle ground between efficient parallel inference of Transformers and accuracy of autoregressive decoding.
        - Make k output layers for k-step-ahead future prediction.
        - First result that outperforms the base Transformer while being computationally more efficient (with small k’s).
    - **Adversarial Text Generation via Feature-Mover's Distance**
        - Another generation method that makes use of the *features*. (Yoon Kim’s CNN features.)
        - A WGAN-style adversarial training in the feature space.

- Other notable papers
    - Binary Rating Estimation with Graph Side Information
    - Hamiltonian Variational Auto-Encoder
    - Exponentiated Strongly Rayleigh Distributions
    - Sparsified SGD with Memory
    - Adding One Neuron Can Eliminate All Bad Local Minima
    - MetaGAN: An Adversarial Approach to Few-Shot Learning
    - Deep Generative Models with Learnable Knowledge Constraints
    - End-to-End Differentiable Physics for Learning and Control
    - Mean-field theory of graph neural networks in graph partitioning
    - Learning in Games with Lossy Feedback
    - Sketching Method for Large Scale Combinatorial Inference
    - MixLasso: Generalized Mixed Regression via Convex Atomic-Norm Regularization
    - L1-regression with Heavy-tailed Distributions
    - Forward Modeling for Partial Observation Strategy Games - A StarCraft Defogger
    - M-Walk: Learning to Walk over Graphs using Monte Carlo Tree Search

---

## GILBO: One Metric to Measure Them All

*Wed Dec 5th 03:45 -- 03:50 PM @ Room 220 E*

- Proposes a lower bound of the mutual information as a metric for deep generative models.
    - “Generative Information Lower BOund”: I(X; Z)
    - ...instead of the log-likelihood (and its lower bound ELBO)
    - Related to the log of effective description length.
    - Entirely independent of the true data. (←→ FID)
    - Requires a tractable prior such that the encoder e(z|x) can be computed. (E.g., VAEs and GANs)

## On the Local Minima of the Empirical Risk

*Wed Dec 5th 04:10 -- 04:15 PM @ Room 517 CD*

(From Rong Ge & Michael Jordan's group.)

## Recurrent World Models Facilitate Policy Evolution*

*Wed Dec 5th 04:25 -- 04:40 PM @ Room 220 CD*

- A MDN-RNN-based RL agent that “dreams” in its world model, learned using a VAE.
- Interesting performances on car racing and VizDoom experiments.

## DAGs with NO TEARS: Continuous Optimization for Structure Learning

*Wed Dec 5th 04:40 -- 04:45 PM @ Room 220 E*

- A smooth characterization of the DAG constraint is available!
- Implementable within ~50 lines with existing solvers.
- Allows a smooth, efficient & no-tears methods for structure learning!

## GPyTorch: Blackbox Matrix-Matrix Gaussian Process Inference with GPU Acceleration

- [https://gpytorch.ai/](https://gpytorch.ai/)
- Much faster Gaussian Process (GP) learning that exploits GPU acceleration!
- Entirely modular based on PyTorch.

---

## Wednesday Poster Session B (05:00 PM — 07:00 PM)

170 papers

(Lots of interesting posters!)

- Noteworthy
    - **Isolating Sources of Disentanglement in Variational Autoencoders**
        - (David Duvenaud masterfully presents the work while taking good care of his baby!)
        - Argues that the claims of disentanglement given by beta-VAEs are unsubstantialized
        - Decomposes the KL penalty into three terms (assuming factorized posterior):
            - Index-Code Mutual Information: how much the latent distribution remembers training samples
            - Dimension-wise KL: mainly a regularization term between prior and posterior, dimension-wise
            - Total Correlation: How entangled the d-dimensional posterior is.
        - Only using TC as the penalty provides a plug-in replacement that gives better disentanglement.
        - David Duvenaud (holding a baby): Still unclear what the correct definition of disentanglement should be.
    - **Generative Neural Machine Translation**
        - A generatve NMT model with a latent semantic variable z: x→y with both z→x and z→y
        - Approximate posterior q(z|x, y) is computed with variational inference.
        - Latent variable z handles the semantic commonalities (and syntactic, if two languages are similar) between the two sentences.
    - **Efficient Algorithms for Non-convex Isotonic Regression through Submodular Optimization**
        - (Solo paper by Francis Bach!)
        - Theoretical analysis and application of submodular optimization to isotonic regression.
        - Still “sub-optimal” error bounds.
    - **Simple random search of static linear policies is competitive for reinforcement learning**
        - (The famous sanity check paper from Ben Recht’s group.)
        - A linear random search method can outperform sophisticated deep RL methods such as TRPO
            - Augmented random search (ARS): matrix multiplication + Gaussian-normalized input
        - The variability in performance with even the random search method is huge. Fails badly in 3-40% of the trials.
            - Can’t even test this much with more sophisticated methods, which require much more computation.
        - Both the methodology and performance measurements must be reconsidered in (continuous-time) deep RL.
    - **Assessing Generative Models via Precision and Recall**
        - Proposes Precision and Recall as better alternatives to Inception scores and PID scores
        - Experiments with subsets of MNIST with 1~9 classes included → P/R should be separated, existing measures don’t care
        - Experiments with 80 GAN and VAE variants → GANs tend to have lower recall (mode collapse), VAEs tend to have lower precision (blurry images)
    - **How Much Restricted Isometry is Needed in Nonconvex Matrix Recovery?**
        - Very much. (Strong RIP required.)

- Other notable papers
    - Learning Attentional Communication for Multi-Agent Cooperation
    - Regret bounds for meta Bayesian optimization with an unknown Gaussian process prior
    - Maximum Causal Tsallis Entropy Imitation Learning
    - Learning to Play with Intrinsically-Motivated, Self-Aware Agents
    - Dirichlet belief networks for topic structure learning
    - Graph Convolutional Policy Network for Goal-Directed Molecular Graph Generation
    - Memory Augmented Policy Optimization for Program Synthesis and Semantic Parsing
    - Maximizing Induced Cardinality Under a Determinantal Point Process
    - Are ResNets Provably Better than Linear Predictors?
    - Adaptive Methods for Nonconvex Optimization
    - Long short-term memory and Learning-to-learn in networks of spiking neurons
    - Reducing Network Agnostophobia
    - The emergence of multiple retinal cell types through efficient coding of natural movies
    - Policy Optimization via Importance Sampling
    - Occam's razor is insufficient to infer the preferences of irrational agents
    - Learning to Navigate in Cities Without a Map
    - Negotiable Reinforcement Learning for Pareto Optimal Sequential Decision-Making

---

(Didn’t have time to attend these demos, but they still sound interesting.)

## A Cooperative Visually Grounded Dialogue Game with a Humanoid Robot (Demo)

## BigBlueBot: A Demonstration of How to Detect Egregious Conversations with Chatbots (Demo)

IBM Research AI.

---

# Conference Day 3 (2018.12.06)

## Making Algorithms Trustworthy: What Can Statistical Science Contribute to Transparency, Explanation and Validation?

*Thu Dec 6th 08:30 -- 09:20 AM @ Rooms 220 CDE*

Invited Talk.

- 4 Phases of Evaluation (Analog in Phramaceuticals)
    - Digital Testing (Safety): test-case performance
    - Laboratory Testing (Proof-of-Concept): comparison with humans, user testing
    - Field Testing (Randomized Controlled Trials): controlled trials of impact
    - Routine Use (Post-marketing Surveillance): monitoring for problems
- Ranking the performances of algorithms
    - **Need bootstrap samples (of the test set) for fair comparison among algorithms.**
    - (Only able to evaluate the performance of the exact algorithm you used including SGD — variability among algorithm’s different runs is a whole different matter.)
    - *“Just because something looks the best, it’s not necessarily [the case that] it is the best. We know that for football teams, we know that for schools, we know that for everything – we should be doing the same for algorithms.”*
- What are the good criteria for interpretability?
    - “Transparency does not automatically imply interpretability” (e.g. a complicated decision tree)
    - Checklist(s)
- Explaining Uncertainty
    - You need to acknowledge that your predictions may be uncertain, i.e. “we don’t know!”
    - BUT: if you keep telling the general public that “we don’t know”, they don’t trust you!
    - **“Confident uncertainty” (“muscular uncertainty — we don’t know!”) does not reduce trust in the source.**
- Fairness
    - Example: “What is your heart age?” Lung age, brain age, etc.
        - Phase 3: people who knew their heart age improved their behavior.
        - Colleagues (a group of middle-aged white men): Didn’t care / got annoyed! Why?
            - Nearly everyone had their heart age greater then their actual age.
            - But they don’t wanna do exercise!
        - ...but Spiegelhalter's group were the ones who did it! Based on a rigorous statistical analysis on 2.3 million people.

---

## Learning with SGD and Random Features

*Thu Dec 6th 09:45 -- 09:50 AM @ Room 220 CD*

- Combines SGD with random features (last year’s Test of Time Award recipient!) to provide:
    - Optimal accuracy
    - Minimum computation

## Boosting Black Box Variational Inference

*Thu Dec 6th 09:50 -- 09:55 AM @ Room 220 E*

- Inference as an iterative procedure
- Residual ELBO (RELBO): ELBO + Residuals
    - The next component should be a good approximation of theposterior
    - But should be different from our current approximation!

## Implicit Reparameterization Gradients

*Thu Dec 6th 10:00 -- 10:05 AM @ Room 220 E*

Mikhail Figurnov, Shakir Mohamed, and Andriy Minh

- Explicit vs. Implicit Reparametrization
    - Explicit requires the gradient of the inverse CDF, while implicit requires the gradient of the CDF itself.
    - Implicit allows to decouple sampling from gradient estimation!
    - Implemented in TF library.
    - Experiments 2d-latent VAE on MNIST & LDA.
- **Move away from making modeling choices for computational convenience!**

## Variational Inference with Tail-adaptive f-Divergence

*Thu Dec 6th 10:05 -- 10:20 AM @ Room 220 E*

- Typical VI models under-estimate uncertainty for multi-modal distributions.
- Gives a very-easy-to-implement alternative based on the tail-adaptive f-divergence.
- Moving beyond KL divergences
    - Alpha-divergence generalizes KL and promotes *mass-covering* (if alpha is large).
    - Unfortunately, large alpha causes large variance or even infinite mean!
- Taming the Fat Tail
    - Instead of alpha, estimate the tail probability and use that.
    - This leads to the use of f-divergence, which generalizes alpha-divergence.
    - Computing only a general gradient formula is enough to make use of the f-divergence.
- Practice
    - Works much better with multi-modal distributions. (E.g. 10-d GMMs)
    - Applications to standard datasets as well as continuous-time RL: **Soft Actor-Critic**.

## Why Is My Classifier Discriminatory?

*Thu Dec 6th 10:20 -- 10:25 AM @ Room 220 CD*

- Why does my classifier yield starkly different performances across race, gender, etc.?
    - Error = Bias + Variance + Noise
        - Error from variance: need more samples in that class.
        - Error from bias: need a more expressive model.
        - Error from noise: need more features.
- Need to control each error factor *separately*!
    - Experiments on mortaility prediction.
- Need to take into account **both the data and the model**.

## Human-in-the-Loop Interpretability Prior

*Thu Dec 6th 10:25 -- 10:30 AM @ Room 220 CD*

- Bias the model to be interpretable by human feedback as a Bayesian prior.
- Human-in-the-loop interpretability: update model ←→ human evaluation
    - Prior evaluation is hard, because it involves user studies.
- Step 1: Identify Diverse, High Likelihood Models
- Step 2: Bayesian Optimization with User Studies

## Link Prediction Based on Graph Neural Networks

*Thu Dec 6th 10:30 -- 10:35 AM @ Room 220 CD*

- Given an incomplete network, predict whether an edge should be connected.
- Popular heuristic ones exist: common neighbors, preferential attachment, ...
- SEAL framework
    - Can learn all first-order and second-order heuristics automatically!
    - High-order heuristics can also be learned (provably) with a small subgraph!

## Realistic Evaluation of Deep Semi-Supervised Learning Algorithms

*Thu Dec 6th 10:35 -- 10:40 AM @ Room 220 CD*

- Semi-supervised learning matters in (relatively) small-size data compared to the larger data manifold.
- Evaluation
    - should be compared against many other methods (including transfer learning)
    - across varying label sizes
    - Label mismatch (different classes in train and test)
    - Large enough validation sets

## Automatic Differentiation in ML

*Thu Dec 6th 10:40 -- 10:45 AM @ Room 220 CD*

- Myia: Best of Both Worlds
    - Ahead-of-time, Custom runtime, Pythonic interface, Functional IR representation, Pythonic generality
    - Functional programming frameworks (including recursion)

---

## Thursday Poster Session A (10:45 AM — 12:45 PM)

169 papers

- Noteworthy
    - **Deep State Space Models for Unconditional Word Generation**
        - A probabilistic model for sequences of tokens using generative and inferential flows.
        - “Unconditional”: Generate token sequences from a sequence of standard Normal noise vectors.
        - Variational inference is used. (Not exact inference like Glow or RealNVP)
        - Small-scale experiments against unigram word models.
        - The number of hidden states is around 8~32.
        - Gradually interested in non-AR modeling... but too simplistic.
    - **Towards Deep Conversational Recommendations**
        - Combines a dialog-level RNN with a recommender system using crowdsourced dialog data (~200k) on movies.
        - End-to-end training.
        - (Work by Microsoft, but data collection was apparently supported by Google.)
    - **Gaussian Process Prior Variational Autoencoders**
        - An alternative for dynamic modeling with VAEs by incorporating a Gaussian process.
    - **Multilingual Anchoring: Interactive Topic Modeling and Alignment Across Languages**
        - Matching the topic distributions across languages.

- Other notable papers
    - Implicit Bias of Gradient Descent on Linear Convolutional Networks
    - How Many Samples are Needed to Estimate a Convolutional Neural Network?
    - Nonlocal Neural Networks, Nonlocal Diffusion and Nonlocal Modeling
    - FastGRNN: A Fast, Accurate, Stable and Tiny Kilobyte Sized Gated Recurrent Neural Network
    - Connectionist Temporal Classification with Maximum Entropy Regularization
    - Learning and Inference in Hilbert Space with Quantum Graphical Models
    - Theoretical guarantees for EM under misspecified Gaussian mixture models
    - Wasserstein Variational Inference
    - Boosting Black Box Variational Inference
    - Meta-Learning MCMC Proposals
    - Posterior Concentration for Sparse Deep Learning
    - Graphical model inference: Sequential Monte Carlo meets deterministic approximations
    - Probabilistic Model-Agnostic Meta-Learning
    - Understanding Batch Normalization
    - Overfitting or perfect fitting? Risk bounds for classification and regression rules that interpolate
    - Entropy and mutual information in models of deep neural networks
    - The committee machine: Computational to statistical gaps in learning a two-layers neural network
    - Connecting Optimization and Regularization Paths
    - Large Margin Deep Networks for Classification
    - Dual Swap Disentangling
    - Byzantine Stochastic Gradient Descent

---

## Hyperbolic Neural Networks

*Thu Dec 6th 03:30 -- 03:35 PM @ Room 220 E*

- How do we use Euclidean operations (e.g. distances) in the hyperbolic space?
    - Connect Gyro-vectors and Riemannian hyperbolic geometry
- How do we maintain hyperbolic geometry when modeling with neural networks?
    - A derivation of hyperbolic feed-forward and recurrent networks.
- Website: hyperbolicdeeplearning.com

## Norm matters: efficient and accurate normalization schemes in deep networks

*Thu Dec 6th 03:35 -- 03:40 PM @ Room 220 E*

Presented by Elad Hoffer

- On Normalization
    - Batch normalization has been successful, but it requires independent samples.
    - BN achieves norm invariance.
    - Bounded Weight Normalization: improve WeightNorm by forcing norm invariance.
    - **L1-BN**: replace the normalization by BN by L2 with L1 (with constant scaling). Easier computation!
        - Successfully trained ResNet completely in half-precision.

## Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels

*Thu Dec 6th 04:55 -- 05:00 PM @ Room 220 E*

---

## Thursday Poster Session B (05:00 PM — 07:00 PM)

- Noteworthy
    - **Learning Beam Search Policies via Imitation Learning**
        - Beam search is universally used, but it is *not* a trained procedure (i.e. unused during training)!
        - Presents a meta-algorithm using imitation learning that learns beam search trajectories during training.
    - **Learning to Multitask**
        - Presents a framework for systematically performing multitask learning.
        - Applications to image classification (multiple image tasks/labels) and news classification (multiple text classification problems).

- Other notable papers
    - Constrained Graph Variational Autoencoders for Molecule Design
    - Towards Understanding Learning Representations: To What Extent Do Different Neural Networks Learn the Same Representation
    - Learning to Reconstruct Shapes from Unseen Classes
    - Wavelet regression and additive models for irregularly spaced data
    - ResNet with one-neuron hidden layers is a Universal Approximator
    - Neural Architecture Optimization
    - Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels
    - Nonparametric Density Estimation under Adversarial Losses
    - Stochastic Chebyshev Gradient Descent for Spectral Optimization
    - Decentralize and Randomize: Faster Algorithm for Wasserstein Barycenters
    - A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks
    - Compact Representation of Uncertainty in Clustering
    - Support Recovery for Orthogonal Matching Pursuit: Upper and Lower bounds
    - Bilevel learning of the Group Lasso structure
    - Sparse PCA from Sparse Linear Regression
    - When do random forests fail?

---

# Workshop Day 1 (2018.12.07)

## The second Conversational AI workshop — today's practice and tomorrow's potential

- (Mostly goal-oriented focus.)

## Competition Track: **The Conversational Intelligence Challenge 2 (ConvAI2)**

- Includes winning presentation from Hugging Face on the persona-based conversation challenge.
- **Lost in Conversation: Human Evaluation Winner!**
    - Learn NLG using OpenAI GPT
    - Persona-enhanced multi-head attention
    - *4/4 Human evaluation score!*
- **Hugging Face: Perplexity Winner!**
    - Learn NLG using OpenAI GPT (best model at the time)
    - Concatenate persona, previous utterance, other person’s utterance — with three separate “dialog state” embeddings!
    - Multi-tasking with the LM loss and an adversarial training loss.
    - *Asks too many questions! Didn’t end up winning human evaluation.*
- Other top submissions included:
    - Seq2seq RNNs
    - GloVe + ULMFiT
    - Auxiliary LM loss on the encoder
    - Independent training of the encoder and the decoder
    - Input-output tying, Mixture of Softmaxes, label smoothing, LOTS of regularization techniques (Meils et al. 2018 + many others)
    - HyperOpt
    - Rule-based (“Constrained”) decoding to prevent repetitions.
        - Paraphrases are a real deal! Repetitions of non-stopwords are tough to detect.
- Conclusions & Discussion (by Jason Weston)
    - Pre-trained generative Transformers work well!
    - But not just about perplexity — search strategy important too!
    - Automated metrics (PPL, F1) are NOT enough for multi-turn dialog!
    - DialogueNLI (Welleck et al. 2018) could be an interesting solution.
    - Wizard of Wikipedia (Dinan et al. 2018): Another task, with more topic depth
        - Contrasts with PersonaChat, which is just a meet-and-greet task.
        - Transformers seem to copy Wikipedia a lot.
        - Generative models may improve a lot!

## Visually grounded interaction and language: Invited Talk by Angeliki Lazaridou

- **Multi-agent Communication as building block**
    - Motivations
        - Passively learning language from large corpora is NOT satisfying, given that grasping the *meanings* out of them is not doable.
        - Language emergence and evolution
        - Facilitates knowledge transfer and information sharing empowering collective intelligence.
    - Popularized (again) in the deep RL era, BUT this field already existed (e.g. Adaptive behaviour, 2003)
        - “Please do cite work before 2015”
- **Why care about compositionality?**
    - “Smaller blocks that are modular and reusable.”
    - Towards more generalizable and robust communication.
- **Referential games with symbolic and pixel input**
    - Work with symbolic (more ancient language space) and pixel (more realistic space) inputs.
    - How do we measure compositionality?
        - An open-ended question. Two *necessary* (but not sufficient) conditions:
            - How isomorphic are the agents’ and natural language spaces? (*topolographic similarity*)
            - Zero-shot generalization to novel stimuli.
    - Results seem to be something... but very unstable results.
        - Slight changes to the setting breaks all the results.
        - “I spent hours and hours and hours looking at the results and tried to figure out if there are meaningful patterns...”
        - Probing results: representations weren’t even predicting basic properties (floor type, shape, ...)
    - Discussion
        - Discrete or continuous communication?
            - If the research question is language, one should look at discrete representations first.
            - If the question is communication, discrete communication isn’t strictly necessary.
        - Conclusions of compositionality do NOT carry over from symbolic to pixel inputs.
        - Communication protocols are too grounded to the environment.
- **Emergent linguistic communication within self-interested agents**
    - Why care?
        - If we’re interested in modeling human communication, this is undoubtedly the more realistic situation.
        - No agents would have perfectly aligned goals. They will coordinate, but possibly with different goals.
    - Unfortunately, this direction isn’t quite easy:
        - Game theory tells us that, if agents’ interests diverge a lot, rational agents shouldn’t communicate.
        - Language evolution tells us that cooperation may represent an important prerequisite for the evolution of language.
        - Empirical evidence suggests that deep RL toolkits are tough to implement anyway.
    - Effective communication?
        - Example: Two researchers working on similar topics and meeting at a conference.
            - They wouldn’t want to reveal full information.
            - But they would want to communicate to selectively share some information and get feedback.
        - If communication success coincides with task success, it is difficult to disentangle meaning acquisition from use acquisition.
    - Experiment: sequential social dilemmas
        - Games in which agents *need* to communicate to get apples (otherwise they will get none).
        - Give agents intrinsic motivation to communicate *in order to make a causal influence on (i.e. change the behavior of) the other agents!*
        - Don’t need to observe the other agents’ reward!
    - Coming up with an environment where agents have to communicate while being not cooperative.

- (Question) Deception?
    - Haven’t tried, seems hard.
- (Question) Can there be discrete communication in the first place? Aren’t the signals continuous anyway?
    - What they communicate are a sequence of symbols.

***Chat with Lazaridou post-talk**

- (Question) In terms of learning grounded and compositional language learning, how good of a framework are referential games? Is there a multi-turn extension (i.e. beyond Alice-Bob) that could be more useful?
    - Unidirectional referential games aren’t necessarily the best, but are the simplest ones. Yes, humans not only speak/explain but also listen/understand.
    - Doesn’t necessarily mean the multi-turn analog makes it better, though. Other better-motivated directions might be better, like self-interested agents.
- (Question) Thoughts on linguistic priors?
    - Depends on your goal. If you want the natural language that we speak, then sure. But our paper (Lazardiou et al. 2018) also shows that this isn’t so easy even with pixel inputs involving simple boxes. (She didn’t seem too interested in e.g. using an NLM for communication, only saying it could be tried if that’s the goal.)

## Visually Grounded Interaction and Language: Invited Talk by Barbara Landau

A cognitive scientist’s view.

- **Learning simple spatial terms: Core and more**
    - ~80 location-related terms in English (in, on, above, ...), some languages just have 2.
    - E.g. how do babies (or machines) acquire all of these terms? Rules? Use patterns? Grounded learning?
- Proposal: “Core” spatial meanings
    - Core (meaningful uses like representating spatial relations, gravitational forces, ...) uses reflect universals.
    - Non-core (non-meaningful uses of in/on/above/...) uses embody variation.
    - **Hypothesis: The two uses can be distinguished by their distributional patterns in usage!**
- Division of labor 1
    - Geometric, rooted in navigation (above/below, east/west, ...).
    - Functional (focus of this talk)
- Division of labor 2: Adpositions still best, verbs do the rest.
- *in*: containment vs. support vs. something else
    - 6 sub-types of containment: loose (full, partial), tight (full, partial), interlock, embed
    - 5 sub-types of support: gravitation (“Core” usage), adhesion, embed, hanging, point attachment
    - Experiment with humans: make humans describe an image (“canonical exemplars”) from one of the sub-types. See how they describe it.
        - Adults and four-year-olds have the same usage patterns across these sub-types!
        - Lexical (non-BE) verbs (“lying in” instead of “is in”): four-year-olds don’t use them, for adults the patterns are still similar.
        - Also similar patterns in different languages.
        - Conclusion: **Core usage is significantly more common.**
    - Lexical verbs (“Adult phrases”): lying in, hanging on, sitting on, attached on, ...
        - Kids do know the phrases, but they don’t use them! Adults do.
- So... what do we mean by “where” for containment and support?
    - Containment: may be only core cases e.g. full, loose
    - Support: may be only core case i.e. “gravitational” support.
- Amending the proposal: What, where, and how
    - Location is expressed through spatial adpositions (where) that define location of figures and grounds (what).
    - BUT: the *how* part expressed by lexical verbs also matters.
- Note from second language learners: Tremendous difficulty *outside* the Core usages!
    - Women in canoe, people on flight, crack in road, yellow line on the road, ...
    - These would have to be learned by “reading the book” (i.e. distributional patterns), but they’re not the Core.
- Conclusions: Rethinking the Problem
    - Sparse representation fo “where”
    - Rich representations of “what” and “how”
    - Core representations should emerge early and be cross-lingually aligned.
    - Non-core often include representations of “how”
- Final word: do we need the baby or the robot? Both!

## Visually Grounded Interaction and Language: Invited Talk by Chris Manning

- **VQA benchmarks are problematic!**
    - Not representative of visual language understanding (inconsistency, ungroundedness, no compositionality, no reasoning, ...)
    - Datasets are limited: artificial images, small space of possible objects and attributes, memorizable.
- **The GQA dataset**
    - Compositional question answering over real-world images grounded on scene graphs.
    - Questions are generated :( build using 500 probabilistic patterns, use paths to create unambiguous multi-step object references
    - Based on Visual Genome, words are cleaned up (constrained large world ontology), scene graphs normalized, ...
    - Unbiasing of the dataset while keeping the general trend (softening the dataset).
    - Questions are perhaps less human-like, but usually more objective (if reasoning can be done).
- **MAC (Memory, Attention, and Composition) Networks**
    - Decomposes a problem into a sequence of explicit reasoning steps!
        - Control state: reasoning operation as attention-based average of a query
        - Memory: retrieved information as attention-based average of the knowledge base (image)
    - The MAC network is a sequence of the 2-component MAC cells
- **Evaluation of MAC on GQA**
    - 80+% Humans, 50+% MAC & Bottom-up, 40+% language-only LSTM
        - “Maybe MAC isn’t as good as we claimed based on the CLEVR dataset”
    - Consistency is also not-so-good across systems.
    - Validity is high among all systems, but *plausibility* (common sense) is bad.
- **Towards reasoning & intelligence**
    - (Biased view) The SQuAD dataset isn’t perfect, but it has been the definitive task during 2016-2018.
        - Manning initially opposed the idea of Turkers generating questions *knowing* the answer already.
        - It did hit the sweet spot. Good progress has been made, and is possibly extendable to other applications.
    - In VQA, such a dataset doesn’t seem to exist, but GQA could be one!
- **The grounding question**
    - I’m not a philosopher, but...
        - The more you think about it, it becomes a very confusing question!
        - Meaning as denotation. Yes. But...
    - But the problem becomes circular if you start incorporating multi-modal signals!
        - Wittegenstein’s use-based representation could also be grounding. (???)
        - Words as other words is ungrounded, yes.
        - But if you take images that correspond to words anyway, is it any different?
        - Semantic parsing is grounding natural language to formal languages? Could be just translation between two languages though...
    - There’s a lot of difficult problems hiding behind this problem.
    - Not saying multi-modal learning is not meaningful of course, but the framework of grounding could be in question.

## Visually Grounded Interaction and Language: Panel Discussion

- How can you tell whether the machine is actually reasoning vs. remembering patterns?
    - Manning: by experiments.
- Humans are so good at this. What are the machines missing? Intentions?
    - We just need more data.
    - Lots of things you can’t find on the web, in a corpus, physics, ...
    - The way that we train our models is not good. We don’t have separate train/test sets.
    - Higher levels of reasoning.
    - Need more algorithms that work better with *less data*. Need more unsupervised or self-supervised learning.
    - Common sense, world knowledge, ...
- Common sense
    - There’s no way to learn common sense from a dataset.
    - Manning: depends on what kinds. Starting points could be some sort of embodied learning. Corpus does include some common sense, no?
    - Deeper common sense principles (from cognitive science view): can two objects occupy the same space at the same time?
        - Do you focus on tasks? Or do you try to take these things into account?
        - Well, as empiricists, humans do learn them from data but just a lot better.
- Cognitively, how do symbols change? Can categories change over time?
    - Representational grounding → more fundamental.
    - Referential grounding: just objects → cultural?
    - They already evolve, upon how you understand a symbol (e.g. atom)
- How much is it okay to tamper with the natural (e.g. Zipfian) distribution?
    - Lazaridou: the Zipfian distribution is more of a feature than a bug. It’s fine to keep it in the training step. BUT what happens in test time? Could be related to the problem of generalization vs. memorization. It’s a problem with machine learning, not the data.
    - Manning: great question, no good answer. Natural distribution is supposed to be good, but the statistics of natural language is so strong. One choice would be uniform (but extreme), and another would be not changing it at all.
    - Test sets should be different and diverse, as Percy Liang claimed. Multiple test sets should be used.
- All these datasets are still constructed artifacts, right? They’re not just those that are “out there in the world.”
    - [http://vizwiz.org/data/](http://vizwiz.org/data/)
- Takeaways
    - (Landau) The symbolic grounding problem is puzzling. Is this an important problem in search of what we want (from visual dialog systems)? If we want practical systems, the grounding problem might not be so critical.
    - (Lazaridou) Synthetic datasets: solution or problem?
    - (Mottaghi) We need machines to learn from interaction. And few examples.
    - (Chai) Grounding: what is it? It’s a loaded word used in multiple contexts (symbolic vs communication grounding).
    - (Kiela) I’m interested in grounding because I’m interested in better natural language understanding. It is only one problem, but it is intimately related to NLU. Especially from the interactive & multi-agent environments.
    - (Manning) (We should solve NLU of course.) Moving into the direction of (visual) intelligence from QA is a great research direction. Across datasets, not just on one dataset.
- What is the definition of common sense that these visual dialog systems should care about? Of course we’ll probably be able to find adversarial examples (be it numerosity, relational reasoning, physics laws) where the system fails. But what are the common senses that we can or should focus on?
- Is the grounding problem (or the goal of language grounding) over-emphasized? If your goal is to build a system that understand language or multi-modal queries or common sense or physical laws, is grounding is just a confusing and circular goal that we should focus less on?

## Other Concurrent Workshops

- Deep Reinforcement Learning
- Causal Learning
- Continual Learning
- All of Bayesian Nonparametrics (Especially the Useful Bits)
- Critiquing and Correcting Trends in Machine Learning
- Bayesian Deep Learning

---

# Workshop Day 2 (2018.12.08)

## Wordplay: Reinforcement and Language Learning in Text-based Games

- BabyAI: First Steps Towards Grounded Language Learning With a Human In the Loop (Invited Talk) - Maxime Chevalier-Boisvert
- **Harnessing the synergy between natural language and interactive learning, Karthik Narasimhan (Princeton)**
    - NLP & RL: separately made lots of progress in recent years.
    - Learning natural language instructions
        - Burst of work in around 2010.
    - Text-based games: the opportunities
        - Learn task-optimized tesxt representations
        - In-game rewards provide unstructured feedback
    - Small game experiment
        - LSTM-DQN (simplest type) gets near-perfect performance on the game.
        - But what are the learned semantics?
            - Nicely clustered word embeddings: “garden” cluster with “tree”, ...
            - Learns similar instructions.
    - The other way: Can we use language to help reinforcement learning?
        - [https://arxiv.org/pdf/1708.00133.pdf](https://arxiv.org/pdf/1708.00133.pdf)
        - Deep RL: huge success in games in general, but they require lots and lots of episodes!
        - Multiple game scenarios: policy transfer is challenging. Currently, each new game requires learning again from scratch.
        - Why is transfer so hard?
            - States and actions are different. Requires learning mappings (sometimes requires as much as data as those required to learn the policy itself).
            - Mapping would require anchors between states. (Hard to get them, of course.)
            - Why don’t we use *text* as anchors?
            - Natural language as anchors between games
            - “Scorpion chases you” vs. “Spiders are chasers”: avoid them!
            - [Game 1] ← Language → [Game 2]
            - Bootstrap learning through text: transfer from one episode (game 1) to another (game 2) in language.
                - State + Description → Embedding.
                - Differentiable Value Iteration. Parameter Learning via loss minimization (Bellman).
            - Ground the semantics of text to *t*he dynamics of the world/environment. NOT to the policy. Can complement other grounding approaches!
                - Incorporating the text already helps. Text + the model-based value iteration network works much much better.
        - What does grounding text to dynamics have to do with text-based games?
            - Generally applicable to many environments
            - Handle global dynamics or pixel-based observations
            - In text-based games, both the state observations and dynamics descriptions can be in text form!
        - **Language can accelarate learning.**
- Humans and models as embodied dialogue agents in text-based games (Contributed Talk) - Jason Weston
- *Wordplay Competition Announced! “First TextWorld Problems”*

## Emergent Communication Workshop

- Spotlights
    - **Countering Language Drift via Grounding**
        - Jason Lee, Kiela, Cho.
    - **Emergent communication via model-based policy transmission**
        - What are the properties of the environment that allows the emergence of compositional & modular language?
        - Existing work: model-free RL, restricted environments, iterated prisoner’s dilemma
        - Rational pedagogy: a model-based, teacher-student framework. What is the best demonstration that the student optimally learns the language?
        - Similar to policy distillation, but primarily interested in what language patterns emerge.
        - Model-Based Policy Transmission: Student (Q-Learner), Teacher acts to optimize student’s Q-values
        - Generalizes to n-step iteration. Reward = Q-value deviation of the student from target.
        - Applicable to generic RL settings: particle world, OpenAI gaze environments, ...
    - **Paying Attention to Function Words**
        - A non-sense-like poem
            - Twas brilliig, and the asdg, ...
            - Content words are meaningless, BUT the function words aren’t always meaningless!
        - Human language evolved to divide the labor between content and function words.
            - Inevitable trivial composition assumption: not enough variablility in *context*.
            - Context-dependent reference: the same object may be the smaller or the larger one!
            - Can agents separate the content (shape/color/...) & function (relative/comparative words?)
            - By adding *attention*, one signal captures the objective content, and another captures function (highest/lowest).
    - **Seq2Seq Mimic Words: A Signaling Perspective**
        - Non-cooperative Environments
            - Not fully aligned interests
            - Independent learners
        - Novel environments inspired by signaling theory
            - Red and Blue have some non-directly observable traits and limitations
            - Gray want to buy the blue robot, but cannot see the robot until paid.
            - All agents start tabula rasa.
    - **Bayesian Action Decoder for Deep Multi-Agent RL**
        - Real-world games that humans find challenging in interpreting (contextually)!
        - Hanabi: you can see everyone else’s cards, but not yours!
            - Public & private features, messaging is costly, cooperative game
            - Bayesian approach: P(hand|action) ~ P(action|hand)P(hand)
            - 10^14 possibilities
        - Results
            - The Bayesian action decoder performs really well (much better than other LSTM-based methods or hard-coded ones) on real game!
            - 40% of the information contained in the action itself!
            - Some previously unknown strategies have been found.
        - Releasing “The Hanabi Challenge: A New Frontier for AI Research” for reproducible & challenging environment!
    - **Intrinsic social motiviation via causal inference in multi-agent RL**
        - Causal framework
            - Put intrinsic reward for being able to causally infer other agents’ actions.
            - Other agent goals not observed (need counterfactuals!)
            - Test on Sequential Social Dilemma
        - Relates to mutual information between agents’ rewards.
        - A high-influence agent: only moves when reward (food) exists. This *action* influences other agents!
        - A communication channel with separate cheap talk (not necessary to use it)
            - Being influenced (i.e. good listeners) gets you higher reward!

## Other Concurrent Workshops

- Relational Representation Learning
- Learning by Instruction
- Interpretability and Robustness in Audio, Speech, and Language
- Integration of Deep Learning Theories
- Workshop on Meta-Learning
- Infer to Control: Probabilistic Reinforcement Learning and Structured Control
