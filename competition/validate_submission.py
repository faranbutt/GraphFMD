import pandas as pd
import sys

def main(pred_path, test_nodes_path):
    preds = pd.read_csv(pred_path, encoding='utf-8-sig')
    test_nodes = pd.read_csv(test_nodes_path, encoding='utf-8-sig')
    preds.columns = preds.columns.str.lower().str.strip()
    test_nodes.columns = test_nodes.columns.str.lower().str.strip()

    if 'id' not in preds.columns:
        print(f"Error: Missing 'id' column. Found: {preds.columns.tolist()}")
        sys.exit(1)
        
    if set(preds['id']) != set(test_nodes['id']):
        sys.exit(1)
    
    print("Validation Successful")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])