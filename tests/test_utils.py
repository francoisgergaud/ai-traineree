import json
import random

import numpy as np
import pytest
import torch
from ai_traineree.utils import (serialize, str_to_list, str_to_number,
                                str_to_seq, str_to_tuple, to_tensor)
from conftest import deterministic_interactions


def generate_value():
    if random.random() < 0.5:
        return [random.random() for _ in range(random.randint(0, 10))]
    return random.random()


def test_to_tensor_tensor():
    # Assign
    t = torch.tensor([0, 1, 2, 3])

    # Act
    new_t = to_tensor(t)

    # Assert
    assert torch.equal(new_t, t)


def test_to_tensor_list():
    # Assign
    int_l = [0, 1, 2, 3]
    float_l = [0.5, 1.1, 2.9, 3.0]
    int_l_shape = [list(range(4*i, 4*(i+1))) for i in range(5)]

    # Act
    int_t = to_tensor(int_l)
    float_t = to_tensor(float_l)
    int_t_shape = to_tensor(int_l_shape)

    # Assert
    assert torch.equal(torch.tensor(int_l), int_t)
    assert torch.equal(torch.tensor(float_l), float_t)
    assert torch.equal(torch.tensor(int_l_shape), int_t_shape)

    assert int_t.shape == (4,)
    assert float_t.shape == (4,)
    assert int_t_shape.shape == (5, 4)


def test_to_tensor_list_tesors():
    # Assign
    int_l = [torch.tensor([0, 1, 2, 3]), torch.tensor([1, 2, 5, 10])]
    float_l = [torch.tensor([0.5, 1.1, 2.9, 3.0]), torch.tensor([0.1, 0.1, 0.1, 10.0])]

    # Act
    int_t = to_tensor(int_l)
    float_t = to_tensor(float_l)

    # Assert
    assert torch.equal(torch.stack(int_l), int_t)
    assert torch.equal(torch.stack(float_l), float_t)

    assert int_t.shape == (2, 4)
    assert float_t.shape == (2, 4)


def test_to_tensor_numpy():
    # Assign
    int_l = np.array([0, 1, 2, 3])
    float_l = np.random.random(4)
    float_l = np.random.random(4)

    # Act
    int_t = to_tensor(int_l)
    float_t = to_tensor(float_l)

    # Assert
    assert torch.equal(torch.tensor(int_l), int_t)
    assert torch.equal(torch.tensor(float_l), float_t)


def test_str_to_number():
    assert 1 == str_to_number("1")
    assert 1 == str_to_number(" 1 ")
    assert 1.5 == str_to_number(" 1.5 ")

    with pytest.raises(ValueError):
        str_to_list("[0, 1")


def test_str_to_list():
    assert [0.1, 0, 0, 2.5, 1] == str_to_list("[0.1, 0, 0, 2.5, 1]")
    assert [0.1, 1] == str_to_list("[0.1, 1]")
    assert [0, 1] == str_to_list("[0,1]")

    with pytest.raises(ValueError):
        str_to_list("[0, 1")
    with pytest.raises(ValueError):
        str_to_list("0, 1]")
    with pytest.raises(ValueError):
        str_to_list("[0, 1)")


def test_str_to_tuple():
    assert (1, 4) == str_to_tuple("(1, 4)")
    assert (1, 4) == str_to_tuple("1, 4")
    assert (1, 4) == str_to_tuple("1,4      ")
    assert (4,) == str_to_tuple("4")

    with pytest.raises(ValueError):
        str_to_tuple("(0, 1")
    with pytest.raises(ValueError):
        str_to_tuple("0, 1)")
    with pytest.raises(ValueError):
        str_to_tuple("[0, 1)")


def test_str_to_seq():
    assert [0.1, 0] == str_to_seq("[0.1, 0]") == str_to_list("[0.1, 0]")
    assert [0, 1] == str_to_list("[0,1]") == str_to_list("[0,1]")
    assert (1, 4) == str_to_seq("(1, 4)") == str_to_tuple("(1, 4)")
    assert (1, 4) == str_to_seq("1, 4") == str_to_tuple("1, 4")
    assert (1, 4) == str_to_seq("1,4        ") == str_to_tuple("1,4      ")
    assert (4,) == str_to_seq("4")

    with pytest.raises(ValueError):
        str_to_seq("(0, 1")
    with pytest.raises(ValueError):
        str_to_seq("0, 1)")
    with pytest.raises(ValueError):
        str_to_seq("[0, 1)")
    with pytest.raises(ValueError):
        str_to_seq("{0, 1}")


def test_serialize_experience_individual():
    from ai_traineree.buffers.experience import Experience

    for key in Experience.whitelist:
        value = generate_value()
        e = Experience(**{key: value})

        ser = serialize(e)
        assert ser == '{"%s": "%s"}' % (key, value)


def test_serialize_experience_all_keys():
    import json

    from ai_traineree.buffers.experience import Experience

    state = {key: generate_value() for key in Experience.whitelist}
    e = Experience(**state)

    # Act
    ser = serialize(e)

    # Assert
    assert json.loads(ser) == {k: str(v) for (k, v) in state.items()}


def test_serialize_buffer_state_list():
    from ai_traineree.buffers import BufferState, Experience

    def generate_exp() -> Experience:
        return Experience(
            state=np.random.random(10).tolist(),
            next_state=np.random.random(10).tolist(),
            action=np.random.random(3).tolist(),
            reward=float(random.random()),
            done=bool(random.randint(0, 8) == 8),
        )

    # Assign
    buffer_size = 1e1
    buffer_state = BufferState(type="Type", batch_size=200, buffer_size=buffer_size)
    buffer_state.data = [generate_exp() for _ in range(int(buffer_size))]

    # Act
    ser = serialize(buffer_state)

    # Assert
    des = json.loads(ser)
    assert set(des.keys()) == set(("type", "buffer_size", "batch_size", "data", "extra"))
    assert len(des['data']) == buffer_size
    assert des["type"] == "Type"
    assert des["extra"] is None


def test_serialize_buffer_state_numpy():
    from ai_traineree.buffers import BufferState, Experience

    def generate_exp() -> Experience:
        return Experience(
            state=np.random.random(10),
            next_state=np.random.random(10),
            action=np.random.random(3),
            reward=random.random(),
            done=(random.randint(0, 8) == 8),
        )

    # Assign
    buffer_size = 1e1
    buffer_state = BufferState(type="Type", batch_size=200, buffer_size=buffer_size)
    buffer_state.data = [generate_exp() for _ in range(int(buffer_size))]

    # Act
    ser = serialize(buffer_state)

    # Assert
    des = json.loads(ser)
    assert set(des.keys()) == set(("type", "buffer_size", "batch_size", "data", "extra"))
    assert len(des['data']) == buffer_size
    assert des["type"] == "Type"
    assert des["extra"] is None


def test_serialize_network_state_actual():
    from ai_traineree.agents.dqn import DQNAgent

    agent = DQNAgent(10, 4)
    deterministic_interactions(agent, 30)
    network_state = agent.get_network_state()

    # Act
    ser = serialize(network_state)

    # Assert
    des = json.loads(ser)
    assert set(des['net'].keys()) == set(('target_net', 'net'))


def test_serialize_agent_state_actual():
    from ai_traineree.agents.dqn import DQNAgent

    agent = DQNAgent(10, 4)
    deterministic_interactions(agent, 30)
    state = agent.get_state()

    # Act
    ser = serialize(state)

    # Assert
    des = json.loads(ser)
    assert des['model'] == DQNAgent.name
    assert len(des['buffer']['data']) == 30
    assert set(des['network']['net'].keys()) == set(('target_net', 'net'))


if __name__ == "__main__":
    test_serialize_network_state_actual()
