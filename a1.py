# PR1: Intelligent Agents and PEAS Representation

class SmartHomeAgent:
    def __init__(self):
        self.light = "OFF"
        self.fan = "OFF"

    def perceive_and_act(self, motion, brightness, temperature):
        print("\n--- Sensor Readings ---")
        print("Motion:", motion)
        print("Brightness:", brightness)
        print("Temperature:", temperature, "°C")

        #decision about light
        if motion and brightness == "DARK":
            self.light = "ON"
        else:
            self.light = "OFF"

        #decision about fan
        if temperature >= 30:
            self.fan = "ON"
        else:
            self.fan = "OFF"

        print("\n--- Agent Actions ---")
        print("Light:", self.light)
        print("Fan:", self.fan)


agent = SmartHomeAgent()

agent.perceive_and_act(True, "DARK", 32)
agent.perceive_and_act(False, "BRIGHT", 25)
agent.perceive_and_act(True, "BRIGHT", 31)