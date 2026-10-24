import part1_embedding
import part2_margins
import part3_unit_circle


def run_part(choice):
    if choice == "1":
        print("Running Part 1-2: 2D embedding space")
        part1_embedding.run()

    elif choice == "2":
        print("Running Part 3-6: SphereFace, CosFace, ArcFace")
        part2_margins.run()

    elif choice == "3":
        print("Running Part 7: unit circle experiment")
        part3_unit_circle.run()

    elif choice == "all":
        part1_embedding.run()
        part2_margins.run()
        part3_unit_circle.run()

    else:
        print("Unknown choice:", choice)


# Program starts here
while True:
    print()
    print("===== MENU =====")
    print("1) Part 1-2: 2D embedding space")
    print("2) Part 3-6: SphereFace, CosFace, ArcFace")
    print("3) Part 7: unit circle experiment")
    print("all) Run everything")
    print("q) Quit")

    choice = input("Your choice: ")
    choice = choice.strip().lower()

    if choice == "q":
        break

    run_part(choice)