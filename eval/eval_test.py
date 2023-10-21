from eval import Evaluator

# load the result pickle
import pickle


# read "eval_frame_results.pkl"

def load_pickle(file_path):
    with open(file_path, "rb") as f:
        return pickle.load(f)
    
eval_frame_results = load_pickle("/tmp/frame_results/eval_frame_results.pkl")
evaluator = Evaluator()
evaluator.eval_frame_results = eval_frame_results
evaluator.compute_results(save_results=False)
