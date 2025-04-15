Day 1 (6:00 a.m. – 1:00 p.m.)

6:00 – 6:15
	•	Welcome & Housekeeping
	•	Quick participant intros (if appropriate).
	•	Outline the 3-day schedule.
	•	Emphasize time blocks and breaks.

6:15 – 6:45
	•	Overview of Generative AI
	•	Introduction to generative AI & its applications.
	•	Basics of generative models, why they matter.
	•	High-level look at different generative model families (GANs, VAEs, autoregressive).

6:45 – 7:00
	•	Short Break (avoid too much talking first thing in the morning)

7:00 – 8:00
	•	Deep Learning Primer
	•	Quick recap of essential deep learning concepts.
	•	Review of neural networks & key architectures.
	•	Explanation of optimization techniques (gradient descent, backpropagation).

8:00 – 8:15
	•	Short Break

8:15 – 9:30
	•	Building Blocks of Generative Models
	•	Probability distributions & sampling techniques.
	•	Latent space & representation learning.
	•	Hands-On Exercise: Implement a simple generative model in Python/TensorFlow (or PyTorch).
	•	Provide partial scaffolding so participants must fill in relevant sections.

9:30 – 9:45
	•	Short Break

9:45 – 11:30
	•	Variational Autoencoders (VAEs)
	•	VAE architecture & how it differs from standard autoencoders.
	•	Training VAEs & generating new samples.
	•	Hands-On Exercise: Build a VAE for image generation & reconstruction.
	•	Focus on implementing the encoder and decoder networks with a latent space sampling step.

11:30 – 12:00
	•	Discussion & Q&A
	•	Address sticking points from the VAE exercise.
	•	Quick summary of Day 1’s main concepts.

1:00 – 2:00
	•	Open Lab / Wrap-Up
	•	Participants can continue or debug the VAE exercise.
	•	You and TAs (if any) roam around for help.
	•	End with a brief look ahead to Day 2 (GANs, LLMs, etc.).

⸻

Day 2 (6:00 a.m. – 1:00 p.m.)

6:00 – 6:15
	•	Morning Recap
	•	Quick review of VAE results.
	•	Preview of Day 2 topics.

6:15 – 7:30
	•	Generative Adversarial Networks (GANs)
	•	Theory behind GANs (generator vs. discriminator).
	•	GAN architecture & training process.
	•	Hands-On Exercise: Train a GAN to generate images (e.g., MNIST or a small image dataset).
	•	Evaluate results (loss curves, sample images).

(If an hour feels too long at 6:15, you can break it into a 30-min lecture and a 30–45-min exercise. For instance, 6:15–6:45 lecture, 6:45–7:30 exercise.)

7:30 – 7:45
	•	Short Break

7:45 – 9:00
	•	Introducing LLMs
	•	What lies behind ChatGPT (conceptual overview).
	•	LLMs as Transformers: the shift from RNN-based to attention-based language models.
	•	Different types of Transformers & real-world tasks.
	•	Famous Transformers (BERT, GPT, T5).
	•	Using Transformers without training (prompt engineering).
	•	The Generative AI project lifecycle vs. traditional ML lifecycle.
	•	Hands-On Exercise: Provide summarization & grammatical corrections with prompt engineering.
	•	Students experiment with different prompts to see how outputs change.

9:00 – 9:15
	•	Short Break

9:15 – 10:30
	•	Introducing Attention
	•	Quick throwback to Word2Vec & seq2seq to show earlier approaches.
	•	Limitations of seq2seq.
	•	Attention à la Bahdanau.
	•	Dot product vs. scaled dot product attention.
	•	Introducing attention in Keras (brief conceptual code snippet).
	•	Hands-On Exercise: Translation with seq2seq + attention (inspired by the original paper).
	•	Provide partial code for the encoder-decoder architecture; participants implement the attention mechanism.

10:30 – 10:45
	•	Short Break

10:45 – 12:15
	•	Transformers
	•	Why seq2seq + attention alone had drawbacks.
	•	Multi-Headed Attention concept.
	•	The Transformer architecture: deeper analysis of each component (embedding, positional encoding, attention layers, feed-forward).
	•	Hands-On Exercise: Create a Transformer from scratch (at least the skeleton).
	•	You might provide the overall class structure & major submodules, leaving some fill-in sections for queries/keys/values, multi-head logic, etc.

12:15 – 1:00
	•	Recap & Discussion
	•	Summarize how we went from VAEs/GANs to Transformers for text.
	•	Quick Q&A on the day’s labs.
	•	Preview Day 3 (Hugging Face, training LLMs, fine-tuning).

⸻

Day 3 (6:00 a.m. – 1:00 p.m.)

6:00 – 6:15
	•	Morning Recap
	•	Check in on the Transformer-from-scratch exercise.

6:15 – 7:30
	•	Hugging Face
	•	Introducing the Hugging Face ecosystem.
	•	Introduction to datasets library.
	•	How to use a model from Hugging Face (inference, pipeline).
	•	How to upload a checkpoint to Hugging Face.
	•	(If time, let folks do a quick “Hello World” Hugging Face pipeline call.)

7:30 – 7:45
	•	Short Break

7:45 – 9:15
	•	Training LLMs: The (Not So) Easy Part
	•	When to train vs. not train your LLM.
	•	Computing difficulties (scale, hardware) & catastrophic forgetting.
	•	Full fine-tuning approach (cost trade-offs).
	•	Hands-On Exercise: Perform full fine-tuning of Flan-T5 and verify “forgetfulness.”
	•	They run it on a small sample dataset to measure changes in performance pre vs. post fine-tune.

9:15 – 9:30
	•	Short Break

9:30 – 10:45
	•	Single-Task vs. Multi-Task Fine-Tuning
	•	Strategies to avoid full fine-tuning (LoRA, adapters, etc. if relevant).
	•	Transfer learning to avoid training from scratch.
	•	Hands-On Exercise: Transfer learning of DistilBERT on sentiment.
	•	Compare single-task approach vs. multi-task or partial fine-tuning.

10:45 – 11:00
	•	Short Break

11:00 – 12:00
	•	Wrap-Up Lab & Demos
	•	Participants finalize exercises (DistilBERT or Flan-T5) and optionally show short results.
	•	Additional Q&A or open discussion.

12:00 – 1:00
	•	Final Q&A & Course Conclusion
	•	Revisit main takeaways from the entire 3-day journey.
	•	Suggestions for next steps and advanced topics (e.g., RLHF, more advanced prompt engineering, advanced HF usage).
	•	Collect feedback from participants.

