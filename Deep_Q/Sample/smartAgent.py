#load the weights from file
agent = Agent(state_size=8,action_size=4,seed=0)
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

for i in range(3):
    state = env.reset()
    img = plt.imshow(env.render(mode='rgb_array'))
    for j in range(200):
        action = agent.act(state)
        img.set_data(env.render(mode='rbg_array'))
        plt.axix('off')
        display.display(plt.gcf())
        display.clear_output(wait=True)
        state,reward,done,_ = env.step(action)
        if done:
            break

env.close()