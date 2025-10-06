import os

def main():
    print("Hello, this is the pipeline skeleton!")
    os.makedirs("datamart/bronze", exist_ok=True)
    os.makedirs("datamart/silver", exist_ok=True)
    os.makedirs("datamart/gold", exist_ok=True)
    print("Created datamart folders: bronze, silver, gold")

if __name__ == "__main__":
    main()
