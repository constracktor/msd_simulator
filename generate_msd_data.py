from simulator.n_linked_msd_simulator import n_linked_msd_simulator

if __name__ == '__main__':
    # Initialize model
    msd = n_linked_msd_simulator("config.json")

    print("# Generate control input #")
    control_input = msd.generate_input()

    print("# Run simulation to generate output #")
    output = msd.simulate(control_input)

    # Load config to store data
    config = msd.get_config("config.json")

    print("# Write data to files #")
    msd.write_data(config["input_file"], control_input)
    msd.write_data(config["output_file"], output)

    # Visualize data in PDF format
    if (config["VISUALIZE"]):
        print("# Visualize data #")
        msd.plot_data(control_input, "input")
        msd.plot_data(output, "output")  

    print("return 0")
