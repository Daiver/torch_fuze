class InputOutputDataset:
    def __init__(self, inputs, outputs):
        assert len(inputs) == len(outputs)
        self.inputs = inputs
        self.outputs = outputs

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.outputs[index]


class InputOutputTransformsWrapper:
    def __init__(self, dataset, transforms_input=None, transforms_output=None):
        self.dataset = dataset
        self.transforms_input = transforms_input
        self.transforms_output = transforms_output

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        inp, output = self.dataset[index]
        if self.transforms_input is not None:
            inp = self.transforms_input(inp)
        if self.transforms_output is not None:
            output = self.transforms_output(output)
        return inp, output
