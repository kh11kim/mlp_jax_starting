# Simple Learning Pipeline using Jax

This repository contains code for a basic learning pipeline using Jax, a numerical computing library for machine learning research.

## Usage

A sample usage is as follows:

1. Set hyperparameters:
```
hp = Hyperparam()
hp.layers = [2, 10, 10, 1]
hp.lr = 0.001
hp.batch_size = 128
```

2. Load data:
```
df = pd.read_csv("circle.csv")
dataset = NumpyDataset(df[["x", "y"]].to_numpy(), df["d"].to_numpy())
train_dataset, val_dataset = train_test_split(dataset, train_size=0.9, shuffle=True)

train_loader = data.DataLoader(
    train_dataset, batch_size=hp.batch_size, shuffle=True, collate_fn=numpy_collate)
val_loader = data.DataLoader(
    val_dataset, batch_size=hp.batch_size, collate_fn=numpy_collate)
```

3. Create model and initialize parameters:
```
model = MLP(hp.layers)
key1, key2 = random.split(random.PRNGKey(0))
x = random.normal(key1, (2,))
params = model.init(key2, x)
```

4. Train model and save checkpoints:
```
tx = optax.adam(learning_rate=hp.lr)
state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
trained_state = trainer(
    state, train_loader, val_loader, l2_loss_fn,
    num_epochs=1000, exp_str=hp.to_str())
```

5. Use trained model:
```
params = load("checkpoint/layers:2_10_10_1,lr:0.001,batch_size:128/990/default")
bind_model = model.bind(params)
bind_model(x)
```

## License
This project is licensed under the terms of the MIT license. 
