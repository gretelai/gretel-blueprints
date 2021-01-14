# Downloading a saved synthetic model

This blueprint does not require Gretel Cloud Project, simply launch the notebook and get started!

When generating synthetic data, you are probably familiar with using our "bundle" interface. Models can take a long time to train and
you may want to save that model for use later to generate more data. As you've seen in other blueprints, this can be done by doing:

```
model.save("my_model.tar.gz")
```

The same `SyntheticDataBundle` class has a factory method that can load an unarchived model and generate data from it.

This blueprint has sample code to download a remotely saved model, decmpress and un-tar it and load it back into the bundle interface.
