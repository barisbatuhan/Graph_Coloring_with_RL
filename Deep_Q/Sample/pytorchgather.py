self.qnetwork_local(state).gather(1,actions)
