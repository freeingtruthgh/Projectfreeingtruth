import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import copy
import os
import matplotlib.pyplot as plt

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self,x):
        return self.block(x)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Policy network: takes in state and outputs continuous action
        self.actor = MLP(state_dim, action_dim, hidden_dim)

    def forward(self, state):
        action = self.actor(state)
        return action

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Critic network: evaluates the quality of a state-action pair Q(s,a)
        self.QNet = MLP(state_dim + action_dim, 1, hidden_dim)

    def forward(self, state, action):
        x = torch.cat([state,action], dim=1)
        q_value = self.QNet(x)
        return q_value

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()

        # Value network: estimates the value of a state for IQL updates
        self.VNet =  MLP(state_dim, 1, hidden_dim)

    def forward(self, state):
        value = self.VNet(state)
        return value

class IQLNet:
    def __init__(self, state_dim, action_dim, hidden_dim=256, actor_lr=3e-4,critic_lr=3e-4, value_lr=3e-4, gamma=0.99, tau=0.005,
                 expectile=0.7, temperature=3.0, device="cpu", linear_expectile=False, step_expectile=False, expectile_start = 0.6, expectile_end=0.9, expectile_step_size=5):
         # Device setup to GPU else cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        self.gamma = gamma  # discount factor
        self.tau = tau      # target network update rate
        self.expectile = expectile      # used for value loss and fixed expectile mode
        self.linear_expectile = linear_expectile    # increase expectile linearly from start to end value
        self.expectile_start = expectile_start
        self.expectile_end = expectile_end
        self.step_expectile = step_expectile        # Schedule ecpectile to increase from start to end every N steps
        self.expectile_step_size = expectile_step_size      # Number of steps, N, to update expectile value
        self.zeta = 1.0         # placeholder this is actually calculated in initialize_expectile_scheduler
        self.temperature = temperature  # used for actor loss

        # Check that only one scheduler is enabled
        if self.linear_expectile and self.step_expectile:
            raise ValueError("Only one expectile scheduler can be enabled at a time.")

        # Actor network
        self.actor = Actor(state_dim,action_dim,hidden_dim).to(self.device)

        # Twin Q networks
        self.q1 = QNetwork(state_dim,action_dim, hidden_dim).to(self.device)
        self.q2 = QNetwork(state_dim,action_dim, hidden_dim).to(self.device)

        # Target Q networks
        self.q1_target = copy.deepcopy(self.q1).to(self.device)
        self.q2_target  = copy.deepcopy(self.q2).to(self.device)

        # Value network
        self.value = ValueNetwork(state_dim, hidden_dim).to(self.device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)
        self.value_optimizer = optim.Adam(self.value.parameters(), lr=value_lr)

    
    def initialize_expectile_scheduler(self, total_epochs):
        if not self.step_expectile:
            return

        num_steps = (total_epochs - 1) // self.expectile_step_size

        if num_steps <= 0:
            self.zeta = 1.0
        else:
            self.zeta = (self.expectile_end / self.expectile_start) ** (1.0 / num_steps)

    def update_expectile(self, current_epoch, total_epochs):
        # Linear expectile scheduler
        if self.linear_expectile:
            progress = current_epoch / max(total_epochs - 1, 1)
            self.expectile = self.expectile_start + progress * (self.expectile_end - self.expectile_start)
            return

        # Step expectile scheduler
        if self.step_expectile:
            step_index = current_epoch // self.expectile_step_size
            self.expectile = self.expectile_start * (self.zeta ** step_index)

            # Clamp to avoid small floating point overshoot
            if self.expectile > self.expectile_end:
                self.expectile = self.expectile_end
            return

    def update_target_network(self, source_net, target_net):
        for source_param, target_param in zip(source_net.parameters(), target_net.parameters()):
            target_param.data.copy_(
                self.tau*source_param.data +(1.0-self.tau) * target_param.data
            )
    def expectile_loss(self, diff):
        # Asymmetric squared loss used for expectile regression in IQL
        # This lets the value network favor higher-return in-dataset actions
        weight=torch.where(diff > 0, self.expectile, 1.0 - self.expectile)
        return weight*(diff ** 2)

    def compute_value_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.q1(states, actions)
            q2 = self.q2(states, actions)
            min_q = torch.min(q1, q2)

        # Train V(s) to match an upper expectile of Q(s, a) for dataset actions
        v = self.value(states)
        diff = min_q - v

        value_loss = self.expectile_loss(diff).mean()
        return value_loss

    def compute_q_loss(self, states, actions, rewards, next_states, terminals):
        with torch.no_grad():
            next_v = self.value(next_states)
            target_q = rewards + self.gamma * (1.0 - terminals) * next_v

        # Train both Q-networks toward the Bellman target
        # target = r + gamma * (1 - done) * V(s_next)
        q1_pred = self.q1(states, actions)
        q2_pred = self.q2(states, actions)

        q1_loss = F.mse_loss(q1_pred, target_q)
        q2_loss = F.mse_loss(q2_pred, target_q)

        q_loss = q1_loss + q2_loss
        return q_loss

    def compute_actor_loss(self, states, actions):
        with torch.no_grad():
            q1 = self.q1(states, actions)
            q2 = self.q2(states, actions)
            min_q = torch.min(q1, q2)

            v = self.value(states)
            advantages = min_q - v

            weights = torch.exp(self.temperature * advantages)
            weights = torch.clamp(weights, max=100.0)

        # Train the actor with advantage-weighted behavior cloning
        # Better dataset actions get larger weights
        predicted_actions = self.actor(states)

        # behavior cloning loss
        bc_loss = ((predicted_actions - actions) ** 2).mean(dim=1, keepdim=True)

        actor_loss = (weights * bc_loss).mean()
        return actor_loss

    def train_step(self, states, actions, rewards, next_states, terminals):
        # Run one full IQL update step:
        # value update, Q update, actor update, then target network update

        # Move batch to the correct device
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        terminals = terminals.to(self.device)

        # -------------------------
        # Value network update
        # -------------------------
        value_loss = self.compute_value_loss(states, actions)

        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()

        # -------------------------
        # Q network update
        # -------------------------
        q_loss = self.compute_q_loss(states, actions, rewards, next_states, terminals)

        self.q1_optimizer.zero_grad()
        self.q2_optimizer.zero_grad()
        q_loss.backward()
        self.q1_optimizer.step()
        self.q2_optimizer.step()

        # -------------------------
        # Actor update
        # -------------------------
        actor_loss = self.compute_actor_loss(states, actions)

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # -------------------------
        # Soft update target Q networks
        # -------------------------
        self.update_target_network(self.q1, self.q1_target)
        self.update_target_network(self.q2, self.q2_target)

        return {
            "value_loss": value_loss.item(),
            "q_loss": q_loss.item(),
            "actor_loss": actor_loss.item(),
        }

class Trainer:
    def __init__(self, model, train_loader, val_loader, epochs, device, tag):
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = epochs
        self.tag = tag

        # Device setup to GPU else cpu
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        if model is None:    
            self.model = IQLNet(self.num_classes)
        else:
            self.model = model

        # set after training, initialize as empty torch Tensors
        self.train_value_loss_vector = torch.zeros(self.epochs)
        self.train_q_loss_vector = torch.zeros(self.epochs)
        self.train_actor_loss_vector = torch.zeros(self.epochs)

        self.val_value_loss_vector = torch.zeros(self.epochs)
        self.val_q_loss_vector = torch.zeros(self.epochs)
        self.val_actor_loss_vector = torch.zeros(self.epochs)

        self.expectile_vector = torch.zeros(self.epochs)

        #  Create folders for checkpoints and results if they don't already exist
        if not os.path.exists("checkpoints"):
            os.makedirs("checkpoints")

        if not os.path.exists("results"):
            os.makedirs("results")

    def validate(self, max_batches=None):
        total_value_loss = 0.0
        total_q_loss = 0.0
        total_actor_loss = 0.0
        num_batches = 0

        for batch_idx, batch in enumerate(self.val_loader):
            if max_batches is not None and batch_idx >= max_batches:
                break

            observations, actions, rewards, next_observations, terminals = batch

            observations = observations.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            next_observations = next_observations.to(self.device)
            terminals = terminals.to(self.device)

            with torch.no_grad():
                value_loss = self.model.compute_value_loss(observations, actions)
                q_loss = self.model.compute_q_loss(observations, actions, rewards, next_observations, terminals)
                actor_loss = self.model.compute_actor_loss(observations, actions)

            total_value_loss += value_loss.item()
            total_q_loss += q_loss.item()
            total_actor_loss += actor_loss.item()
            num_batches += 1

        avg_value_loss = total_value_loss / num_batches
        avg_q_loss = total_q_loss / num_batches
        avg_actor_loss = total_actor_loss / num_batches

        return avg_value_loss, avg_q_loss, avg_actor_loss

    # Handles the outer training loop and loss logging across epochs
    def train(self):
        print(f"Starting training", flush=True)
        print(f"device: {self.device}", flush=True)
        print(f"epochs: {self.epochs}", flush=True)

        best_val_q_loss = float("inf")
        self.model.initialize_expectile_scheduler(self.epochs)

        for epoch in range(self.epochs):
            self.model.update_expectile(epoch, self.epochs)
            print(f"Epoch {epoch + 1}: expectile = {self.model.expectile:.4f}", flush=True)
            self.expectile_vector[epoch] = self.model.expectile

            total_train_value_loss = 0.0
            total_train_q_loss = 0.0
            total_train_actor_loss = 0.0
            num_batches = 0

            for batch_idx, batch in enumerate(self.train_loader):
                observations, actions, rewards, next_observations, terminals = batch

                losses = self.model.train_step(observations, actions, rewards, next_observations, terminals)

                total_train_value_loss += losses["value_loss"]
                total_train_q_loss += losses["q_loss"]
                total_train_actor_loss += losses["actor_loss"]
                num_batches += 1

                # Quick validation every 200 batches
                if (batch_idx + 1) % 200 == 0:
                    quick_val_value_loss, quick_val_q_loss, quick_val_actor_loss = self.validate(max_batches=5)

                    print(
                        f"Epoch {epoch + 1}, Batch {batch_idx + 1} | "
                        f"Train Value Loss: {losses['value_loss']:.6f}, "
                        f"Train Q Loss: {losses['q_loss']:.6f}, "
                        f"Train Actor Loss: {losses['actor_loss']:.6f}, "
                        f"Val Value Loss: {quick_val_value_loss:.6f}, "
                        f"Val Q Loss: {quick_val_q_loss:.6f}, "
                        f"Val Actor Loss: {quick_val_actor_loss:.6f}",
                        flush=True
                    )

                    if torch.cuda.is_available():
                        print(
                            f"GPU Memory: {torch.cuda.memory_allocated() / 1024**2:.2f} MB",
                            flush=True
                        )

            avg_train_value_loss = total_train_value_loss / num_batches
            avg_train_q_loss = total_train_q_loss / num_batches
            avg_train_actor_loss = total_train_actor_loss / num_batches
            self.train_value_loss_vector[epoch] = avg_train_value_loss 
            self.train_q_loss_vector[epoch] = avg_train_q_loss 
            self.train_actor_loss_vector[epoch] = avg_train_actor_loss

            # Full validation at end of epoch
            avg_val_value_loss, avg_val_q_loss, avg_val_actor_loss = self.validate()

            self.val_value_loss_vector[epoch] = avg_val_value_loss
            self.val_q_loss_vector[epoch] = avg_val_q_loss
            self.val_actor_loss_vector[epoch] = avg_val_actor_loss

            # Save chechpoints
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'actor_state_dict': self.model.actor.state_dict(),
                    'q1_state_dict': self.model.q1.state_dict(),
                    'q2_state_dict': self.model.q2.state_dict(),
                    'q1_target_state_dict': self.model.q1_target.state_dict(),
                    'q2_target_state_dict': self.model.q2_target.state_dict(),
                    'value_state_dict': self.model.value.state_dict(),
                    'actor_optimizer_state_dict': self.model.actor_optimizer.state_dict(),
                    'q1_optimizer_state_dict': self.model.q1_optimizer.state_dict(),
                    'q2_optimizer_state_dict': self.model.q2_optimizer.state_dict(),
                    'value_optimizer_state_dict': self.model.value_optimizer.state_dict(),
                    }, f"checkpoints/{self.tag}_checkpoint_epoch_{epoch+1}.pth")
                print (f"Saved checkpoint at epoch {epoch+1}.pth", flush=True)

            # Save best checkpoint and onnx
            if avg_val_q_loss < best_val_q_loss:
                best_val_q_loss = avg_val_q_loss

                torch.save({
                    'epoch': epoch + 1,
                    'actor_state_dict': self.model.actor.state_dict(),
                    'q1_state_dict': self.model.q1.state_dict(),
                    'q2_state_dict': self.model.q2.state_dict(),
                    'q1_target_state_dict': self.model.q1_target.state_dict(),
                    'q2_target_state_dict': self.model.q2_target.state_dict(),
                    'value_state_dict': self.model.value.state_dict(),
                    'actor_optimizer_state_dict': self.model.actor_optimizer.state_dict(),
                    'q1_optimizer_state_dict': self.model.q1_optimizer.state_dict(),
                    'q2_optimizer_state_dict': self.model.q2_optimizer.state_dict(),
                    'value_optimizer_state_dict': self.model.value_optimizer.state_dict(),
                    }, f"checkpoints/{self.tag}_best_checkpoint.pth")

                print (f"Saved new best checkpoint at epoch {epoch+1} with val_q_loss = {avg_val_q_loss:.6f}", flush=True)

                self.save(f"results/{self.tag}_best_actor.onnx")

        return [self.train_value_loss_vector.cpu(), self.train_q_loss_vector.cpu(), self.train_actor_loss_vector.cpu(), self.val_value_loss_vector.cpu(), self.val_q_loss_vector.cpu(), self.val_actor_loss_vector.cpu()]

    def save(self, filename=None):
        if filename is None:
            filename = f"results/{self.tag}_iql_actor.onnx"

        self.model.actor.eval()

        # Get example input
        batch = next(iter(self.train_loader))
        observations, _, _, _, _ = batch

        example_input = observations[0:1].to(self.device)

        # Move to CPU for export
        actor_cpu = self.model.actor.to("cpu")
        example_input = example_input.to("cpu")

        with torch.no_grad():
            torch.onnx.export(actor_cpu, example_input, filename, input_names=["state"], output_names=["action"],
                 dynamic_axes={"state": {0: "batch_size"}, "action": {0: "batch_size"}}, opset_version=18,)

        # move actor back on device
        self.model.actor.to(self.device)

        print(f"Actor exported to {filename}")


    def save_results(self, filename=None):
        if filename is None:
            filename = f"results/{self.tag}_IQL_training_results.npz"

        np.savez_compressed(
            filename,
            train_value_loss=self.train_value_loss_vector.cpu().numpy(),
            train_q_loss=self.train_q_loss_vector.cpu().numpy(),
            train_actor_loss=self.train_actor_loss_vector.cpu().numpy(),
            val_value_loss=self.val_value_loss_vector.cpu().numpy(),
            val_q_loss=self.val_q_loss_vector.cpu().numpy(),
            val_actor_loss=self.val_actor_loss_vector.cpu().numpy(),
            expectile = self.expectile_vector.cpu().numpy(),
        )

        print(f"Saved training vectors to {filename}")

    def evaluation(self):

        epochs = range(1, self.epochs + 1)

        # Value loss plot
        plt.figure()
        plt.plot(epochs, self.train_value_loss_vector.cpu(), linewidth=2, label="Training")
        plt.plot(epochs, self.val_value_loss_vector.cpu(), linewidth=2, label="Validation")
        plt.title("Value Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Value Loss")
        plt.legend()
        plt.savefig(f"results/{self.tag}_value_loss_vs_epochs.png")
        plt.close()

        # Q loss plot
        plt.figure()
        plt.plot(epochs, self.train_q_loss_vector.cpu(), linewidth=2, label="Training")
        plt.plot(epochs, self.val_q_loss_vector.cpu(), linewidth=2, label="Validation")
        plt.title("Q Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Q Loss")
        plt.legend()
        plt.savefig(f"results/{self.tag}_q_loss_vs_epochs.png")
        plt.close()

        # Actor loss plot
        plt.figure()
        plt.plot(epochs, self.train_actor_loss_vector.cpu(), linewidth=2, label="Training")
        plt.plot(epochs, self.val_actor_loss_vector.cpu(), linewidth=2, label="Validation")
        plt.title("Actor Loss vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Actor Loss")
        plt.legend()
        plt.savefig(f"results/{self.tag}_actor_loss_vs_epochs.png")
        plt.close()

        # expectile plot
        plt.figure()
        plt.plot(epochs, self.expectile_vector.cpu(), linewidth=2, label="Expectile")
        plt.title("Expectile vs Epochs")
        plt.xlabel("Epochs")
        plt.ylabel("Expectile")
        plt.legend()
        plt.savefig(f"results/{self.tag}_expectile_vs_epochs.png")
        plt.close()

        print(f"Saved evaluation plots to results/", flush=True)