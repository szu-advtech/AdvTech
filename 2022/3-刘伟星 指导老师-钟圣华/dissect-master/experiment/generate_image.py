from generator_int_experiment import load_model,parseargs

if __name__ == '__main__':
    main()


def main():
    args = parseargs()
    model = load_model(args)
    for i in range(2000):
        image=model.