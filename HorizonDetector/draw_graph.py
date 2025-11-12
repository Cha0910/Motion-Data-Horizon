import os
import pandas as pd
import matplotlib.pyplot as plt

# Base directory containing all scenario folders
base_dir = "../outputs"
xlsx_folder_name = "xlsx"
graph_folder_name = "graph"

input_path = os.path.join(base_dir, xlsx_folder_name)
# 3. GRAPH 출력 경로 설정 (그래프를 저장할 곳)
output_path = os.path.join(base_dir, graph_folder_name)
os.makedirs(output_path, exist_ok=True) # output_path는 이제 graph 폴더를 가리킴

# Process all .xlsx files in xlsx folder
for file_name in os.listdir(input_path):
    if file_name.endswith(".xlsx"):
        file_path = os.path.join(input_path, file_name)

        try:
            df = pd.read_excel(file_path)

            # Roll Graph
            plt.figure(figsize=(10, 5))
            plt.plot(df['Frame No'], df['Roll Data'])
            plt.xlabel('Frame Number')
            plt.ylabel('Roll Data')
            plt.title('Roll Data over Frame Number')
            plt.grid(True)
            plt.savefig(os.path.join(output_path, f"{os.path.splitext(file_name)[0]}_roll.png"))
            plt.close()

            # Pitch Graph
            plt.figure(figsize=(10, 5))
            plt.plot(df['Frame No'], df['Pitch Data'])
            plt.xlabel('Frame Number')
            plt.ylabel('Pitch Data')
            plt.title('Pitch Data over Frame Number')
            plt.grid(True)
            plt.savefig(os.path.join(output_path, f"{os.path.splitext(file_name)[0]}_pitch.png"))
            plt.close()

            print(f"Graphs generated for {file_name} in {output_path}")

        except Exception as e:
            print(f"Error processing {file_name} in {output_path}: {e}")

print("✅ Completed processing all XLSX files.")
