import os
from kerastuner.tuners import Hyperband


def tune(
    dir,
    project_name,
    model,
    train_data,
    val_data,
    max_epochs=10,
    executions_per_trial=3,
    hyperband_iterations=2,
    search_epochs=8,
    verbose=1,
    seed=66,
):
    print("\nTuning\n")
    tuner = Hyperband(
        model,
        objective="val_accuracy",
        max_epochs=max_epochs,
        executions_per_trial=executions_per_trial,
        hyperband_iterations=hyperband_iterations,
        directory=os.path.join(dir, 'logs'),
        project_name=str(project_name),
        seed=seed,
        overwrite=True,
    )
    search_space_summary = tuner.search_space_summary()
    tuner.search(
        train_data, epochs=search_epochs, validation_data=val_data, verbose=verbose
    )
    results_summary = tuner.results_summary()
    best_hps = tuner.get_best_hyperparameters()[0]
    best_hps_config = best_hps.get_config()
    best_model = tuner.hypermodel.build(best_hps)
    # best_model_config = best_model.get_config()
    # best_model_summary = best_model.summary()
    return best_model


def train(model, train_data, val_data, callbacks, train_epochs=10):
    print("\nTraining\n")
    steps_per_epoch = train_data.samples // train_data.batch_size
    validation_steps = val_data.samples // val_data.batch_size
    history = model.fit(
        train_data,
        epochs=train_epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_data,
        validation_steps=validation_steps,
        callbacks=callbacks,
    )
    return history, model


def test(model, test_data):
    print("\nTesting\n")
    results = model.evaluate(test_data, return_dict=True)
    return results