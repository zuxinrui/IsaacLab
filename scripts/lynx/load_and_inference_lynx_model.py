import torch
from rsl_rl.modules import ActorCritic

def load_and_inference_lynx_model():
    """
    Loads a trained PyTorch model, performs inference with a dummy input,
    and prints the output.
    """
    model_path = "/home/zuxinrui/IsaacLab/scripts/lynx/2025-07-28_17-51-38/model_4000.pt"

    try:
        # Load the model
        # Ensure the model is loaded onto the correct device (CPU or GPU)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        loaded_state_dict = torch.load(model_path, map_location=device)

        # Instantiate ActorCritic model
        observation_space_size = 25
        action_space_size = 6
        model = ActorCritic(num_actor_obs=observation_space_size, num_critic_obs=observation_space_size, num_actions=action_space_size, actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128])

        # Move the model to the specified device
        model.to(device)

        # Load the state dictionary into the instantiated model
        model.load_state_dict(loaded_state_dict['model_state_dict'])

        # Create a dummy input tensor
        dummy_input_size = 25
        dummy_input = torch.randn(1, dummy_input_size).to(device)

        # Ensure the model is in evaluation mode
        model.eval()

        # Perform a forward pass (inference)
        with torch.no_grad():
            actions = model.act_inference(dummy_input)

        # Print the output
        print(f"Model loaded successfully from: {model_path}")
        print(f"Dummy input shape: {dummy_input.shape}")
        print(f"Inference output (actions) shape: {actions.shape}")
        print(f"Inference output (actions): {actions}")

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
    except Exception as e:
        print(f"An error occurred during model loading or inference: {e}")

if __name__ == "__main__":
    load_and_inference_lynx_model()