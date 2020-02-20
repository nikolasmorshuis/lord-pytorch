base_config = dict(
	content_dim=128,
	class_dim=256,

	content_std=1,
	content_decay=1e-4,

	n_adain_layers=4,
	adain_dim=256,

	perceptual_loss=dict(
		layers=[2, 7, 12, 21, 30]
	),

	train=dict(
		batch_size=128,
		n_epochs=200,

		learning_rate=dict(
			generator=3e-4,
			latent=3e-3,
			min=1e-5
		)
	),

	train_encoders=dict(
		batch_size=64,
		n_epochs=200,
		learning_rate=1e-4,
		learning_rate_min=1e-5
	)
)
