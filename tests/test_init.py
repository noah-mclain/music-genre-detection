import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_NAME = "src"
print(f"Testing package {PACKAGE_NAME}")


class PackageImportTest:
    @staticmethod
    def run():
        try:
            package = __import__(PACKAGE_NAME)
            print("package imported successfully")
            return package
        except Exception as e:
            print("Failed to import package", e)


class GenreClassiferTest:
    @staticmethod
    def run():
        try:
            from src import GenreClassifier

            genres = [
                "blues",
                "classical",
                "country",
                "disco",
                "hiphop",
                "jazz",
                "metal",
                "pop",
                "reggae",
                "rock",
            ]
            model_path = ""
            classifier = GenreClassifier(model_path, genres)
            print("Genreclassifier instantiation successful")

        except Exception as e:
            print("Genreclassifier instantiation failed", e)


class GTZANDatatestTest:
    @staticmethod
    def run():
        try:
            from src import GTZANDataset

            data_dir = ""
            dataset = GTZANDataset(data_dir)
            sample = dataset[0]
            print("GTZANDatatest insantiation successful")
            print(f"x: {sample[0].shape}, y: {sample[1]}")

        except Exception as e:
            print("GTZANDatatest insantiation failed", e)


def main():
    package = PackageImportTest.run()

    GenreClassiferTest.run()
    GTZANDatatestTest.run()
    print("All tests complete")


if __name__ == "__main__":
    main()
