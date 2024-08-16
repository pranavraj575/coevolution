import pygame, colorsys
from pygame import draw
import numpy as np
from pygame.math import Vector2
from pygame.transform import rotozoom

# Team NEEDS to be imported here to work
from repos.pyquaticus.pyquaticus.envs.pyquaticus import PyQuaticusEnv, Team
from repos.pyquaticus.pyquaticus.structs import RenderingPlayer, Flag, CircleObstacle, PolygonObstacle
from repos.pyquaticus.pyquaticus.base_policies.base import BaseAgentPolicy
from repos.pyquaticus.pyquaticus.utils.utils import mag_heading_to_vec


class MyQuaticusEnv(PyQuaticusEnv):
    """
    keep track of observations of each agent
    """

    def __init__(self, save_video=False, frame_freq=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.obs_record = []
        self.images = []
        self.save_vid = save_video
        self.frame_cnt = 0
        self.frame_freq = frame_freq

    def observation_space(self, agent):
        return self.observation_spaces[agent]

    def action_space(self, agent):
        return self.action_spaces[agent]

    def step(self, *args, **kwargs):
        obs, rewards, terminated, truncated, info = super().step(*args, **kwargs)
        self.obs_record.append(obs)
        if self.save_vid:
            if not self.frame_cnt%self.frame_freq:
                self.images.append(pygame.surfarray.array3d(self._ret_image_only()))

            self.frame_cnt += 1

        return obs, rewards, terminated, truncated, info

    def write_video(self, video_file, fps=30):
        from moviepy.editor import ImageSequenceClip
        real_fps = fps//self.frame_freq  # fps//(self.render_fps*self.frame_freq)
        clip = ImageSequenceClip(self.images, fps=real_fps)
        clip.write_videofile(video_file, fps=real_fps)

    def reset(self, *args, **kwargs):
        thing = super().reset(*args, **kwargs)
        self.obs_record = []
        self.images = []
        self.frame_cnt = 0
        return thing

    def _ret_image_only(self):
        """
        Overridden method inherited from `Gym`.

        Draws all players/flags/etc on the pygame screen.
        """
        if self.screen is None:
            pygame.init()
            pygame.display.set_caption("Capture the Flag")
            if self.render_mode:
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (
                        self.arena_width + 2*self.arena_offset,
                        self.arena_height + 2*self.arena_offset,
                    )
                )
                self.isopen = True
                self.font = pygame.font.SysFont(None, int(2*self.pixel_size*self.agent_radius))
            else:
                raise Exception(
                    "Sorry, render modes other than 'human' are not supported"
                )

        if self.clock is None:
            self.clock = pygame.time.Clock()

        if self.state == {}:
            return None

        # arena coordinates
        if self.border_width%2 == 0:
            top_left = (
                self.arena_offset - self.border_width/2 - 1,
                self.arena_offset - self.border_width/2 - 1,
            )
            top_middle = (
                self.arena_width/2 + self.arena_offset - 1,
                self.arena_offset - 1,
            )
            top_right = (
                self.arena_width + self.arena_offset + self.border_width/2 - 1,
                self.arena_offset - self.border_width/2 - 1,
            )

            bottom_left = (
                self.arena_offset - self.border_width/2 - 1,
                self.arena_height + self.arena_offset + self.border_width/2 - 1,
            )
            bottom_middle = (
                self.arena_width/2 + self.arena_offset - 1,
                self.arena_height + self.arena_offset - 1,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width/2 - 1,
                self.arena_height + self.arena_offset + self.border_width/2 - 1,
            )

        elif self.border_width%2 != 0:
            top_left = (
                self.arena_offset - self.border_width/2,
                self.arena_offset - self.border_width/2,
            )
            top_middle = (self.arena_width/2 + self.arena_offset, self.arena_offset)
            top_right = (
                self.arena_width + self.arena_offset + self.border_width/2,
                self.arena_offset - self.border_width/2,
            )

            bottom_left = (
                self.arena_offset - self.border_width/2,
                self.arena_height + self.arena_offset + self.border_width/2,
            )
            bottom_middle = (
                self.arena_width/2 + self.arena_offset,
                self.arena_height + self.arena_offset,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width/2,
                self.arena_height + self.arena_offset + self.border_width/2,
            )

        # screen
        self.screen.fill((255, 255, 255))

        # arena border and scrimmage line
        draw.line(self.screen, (0, 0, 0), top_left, top_right, width=self.border_width)
        draw.line(
            self.screen, (0, 0, 0), bottom_left, bottom_right, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_left, bottom_left, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_right, bottom_right, width=self.border_width
        )
        draw.line(
            self.screen, (0, 0, 0), top_middle, bottom_middle, width=self.border_width
        )
        # Draw Points Debugging
        if self.config_dict["render_field_points"]:
            for v in self.config_dict["aquaticus_field_points"]:
                draw.circle(self.screen, (128, 0, 128),
                            self.world_to_screen(self.config_dict["aquaticus_field_points"][v]), 5, )

        agent_id_blit_poses = {}

        for team in Team:
            flag = self.flags[int(team)]
            teams_players = self.agents_of_team[team]
            color = "blue" if team == Team.BLUE_TEAM else "red"

            # Draw team home region
            home_center_screen = self.world_to_screen(self.flags[int(team)].home)
            draw.circle(
                self.screen,
                (0, 0, 0),
                home_center_screen,
                self.catch_radius*self.pixel_size,
                width=round(self.pixel_size/10),
            )

            if not self.state["flag_taken"][int(team)]:
                # Flag is not captured, draw normally.
                flag_pos_screen = self.world_to_screen(flag.pos)
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    self.flag_radius*self.pixel_size,
                )
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    (self.flag_keepout - self.agent_radius)*self.pixel_size,
                    width=round(self.pixel_size/10),
                )
            else:
                # Flag is captured so draw a different shape
                flag_pos_screen = self.world_to_screen(flag.pos)
                draw.circle(
                    self.screen,
                    color,
                    flag_pos_screen,
                    0.55*(self.pixel_size*self.agent_radius)
                )

            # Draw obstacles:
            for obstacle in self.obstacles:
                if isinstance(obstacle, CircleObstacle):
                    draw.circle(
                        self.screen,
                        (128, 128, 128),
                        self.world_to_screen(obstacle.center_point),
                        obstacle.radius*self.pixel_size,
                        width=3
                    )
                elif isinstance(obstacle, PolygonObstacle):
                    draw.polygon(
                        self.screen,
                        (128, 128, 128),
                        [self.world_to_screen(p) for p in obstacle.anchor_points],
                        width=3,
                    )

            for player in teams_players:
                # render tagging
                player.render_tagging(self.tagging_cooldown)

                # heading
                orientation = Vector2(list(mag_heading_to_vec(1.0, player.heading)))
                ref_angle = -orientation.angle_to(self.UP)

                # transform position to pygame coordinates
                blit_pos = self.world_to_screen(player.pos)
                rotated_surface = rotozoom(player.pygame_agent, ref_angle, 1.0)
                rotated_surface_size = np.array(rotated_surface.get_size())
                rotated_blit_pos = blit_pos - 0.5*rotated_surface_size

                # blit agent onto screen
                self.screen.blit(rotated_surface, rotated_blit_pos)

                # blit agent number onto agent
                agent_id_blit_poses[player.id] = (
                    blit_pos[0] - 0.35*self.pixel_size*self.agent_radius,
                    blit_pos[1] - 0.6*self.pixel_size*self.agent_radius
                )

        # render agent ids
        if self.render_ids:
            for team in Team:
                teams_players = self.agents_of_team[team]
                font_color = "white" if team == Team.BLUE_TEAM else "black"
                for player in teams_players:
                    player_number_label = self.font.render(str(player.id), True, font_color)
                    self.screen.blit(player_number_label, agent_id_blit_poses[player.id])

        # visually indicate distances between players of both teams
        assert len(self.agents_of_team) == 2, "If number of teams > 2, update code that draws distance indicator lines"

        for blue_player in self.agents_of_team[Team.BLUE_TEAM]:
            if not blue_player.is_tagged or (blue_player.is_tagged and blue_player.on_own_side):
                for red_player in self.agents_of_team[Team.RED_TEAM]:
                    if not red_player.is_tagged or (red_player.is_tagged and red_player.on_own_side):
                        blue_player_pos = np.asarray(blue_player.pos)
                        red_player_pos = np.asarray(red_player.pos)
                        a2a_dis = np.linalg.norm(blue_player_pos - red_player_pos)
                        if a2a_dis <= 2*self.catch_radius:
                            hsv_hue = (a2a_dis - self.catch_radius)/(2*self.catch_radius - self.catch_radius)
                            hsv_hue = 0.33*np.clip(hsv_hue, 0, 1)
                            line_color = tuple(255*np.asarray(colorsys.hsv_to_rgb(hsv_hue, 0.9, 0.9)))

                            draw.line(
                                self.screen,
                                line_color,
                                self.world_to_screen(blue_player_pos),
                                self.world_to_screen(red_player_pos),
                                width=self.a2a_line_width
                            )

        if self.render_mode:
            pygame.event.pump()
            self.clock.tick(self.render_fps)
            pygame.display.flip()

        return pygame.transform.flip(self.screen, False, True)

    def _email_ret_image_only(self):
        """
        Overridden method inherited from `Gym`.

        Draws all players/flags/etc on the pygame screen.
        """

        screen = pygame.Surface(
            (
                self.arena_width + 2*self.arena_offset,
                self.arena_height + 2*self.arena_offset,
            )
        )
        self.isopen = True
        self.font = pygame.font.SysFont(None, int(2*self.pixel_size*self.agent_radius))

        if self.state == {}:
            return None

        # arena coordinates
        if self.border_width%2 == 0:
            top_left = (
                self.arena_offset - self.border_width/2 - 1,
                self.arena_offset - self.border_width/2 - 1,
            )
            top_middle = (
                self.arena_width/2 + self.arena_offset - 1,
                self.arena_offset - 1,
            )
            top_right = (
                self.arena_width + self.arena_offset + self.border_width/2 - 1,
                self.arena_offset - self.border_width/2 - 1,
            )

            bottom_left = (
                self.arena_offset - self.border_width/2 - 1,
                self.arena_height + self.arena_offset + self.border_width/2 - 1,
            )
            bottom_middle = (
                self.arena_width/2 + self.arena_offset - 1,
                self.arena_height + self.arena_offset - 1,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width/2 - 1,
                self.arena_height + self.arena_offset + self.border_width/2 - 1,
            )

        elif self.border_width%2 != 0:
            top_left = (
                self.arena_offset - self.border_width/2,
                self.arena_offset - self.border_width/2,
            )
            top_middle = (self.arena_width/2 + self.arena_offset, self.arena_offset)
            top_right = (
                self.arena_width + self.arena_offset + self.border_width/2,
                self.arena_offset - self.border_width/2,
            )

            bottom_left = (
                self.arena_offset - self.border_width/2,
                self.arena_height + self.arena_offset + self.border_width/2,
            )
            bottom_middle = (
                self.arena_width/2 + self.arena_offset,
                self.arena_height + self.arena_offset,
            )
            bottom_right = (
                self.arena_width + self.arena_offset + self.border_width/2,
                self.arena_height + self.arena_offset + self.border_width/2,
            )

        # screen
        screen.fill((255, 255, 255))

        # arena border and scrimmage line
        draw.line(screen, (0, 0, 0), top_left, top_right, width=self.border_width)
        draw.line(
            screen, (0, 0, 0), bottom_left, bottom_right, width=self.border_width
        )
        draw.line(
            screen, (0, 0, 0), top_left, bottom_left, width=self.border_width
        )
        draw.line(
            screen, (0, 0, 0), top_right, bottom_right, width=self.border_width
        )
        draw.line(
            screen, (0, 0, 0), top_middle, bottom_middle, width=self.border_width
        )
        # Draw Points Debugging
        if self.config_dict["render_field_points"]:
            for v in self.config_dict["aquaticus_field_points"]:
                draw.circle(screen, (128, 0, 128), self.world_to_screen(self.config_dict["aquaticus_field_points"][v]),
                            5, )

        agent_id_blit_poses = {}

        for team in Team:
            flag = self.flags[int(team)]
            print(team)
            print(self.agents_of_team.keys())
            teams_players = self.agents_of_team[team]
            color = "blue" if team == Team.BLUE_TEAM else "red"

            # Draw team home region
            home_center_screen = self.world_to_screen(self.flags[int(team)].home)
            draw.circle(
                screen,
                (0, 0, 0),
                home_center_screen,
                self.catch_radius*self.pixel_size,
                width=round(self.pixel_size/10),
            )

            if not self.state["flag_taken"][int(team)]:
                # Flag is not captured, draw normally.
                flag_pos_screen = self.world_to_screen(flag.pos)
                draw.circle(
                    screen,
                    color,
                    flag_pos_screen,
                    self.flag_radius*self.pixel_size,
                )
                draw.circle(
                    screen,
                    color,
                    flag_pos_screen,
                    (self.flag_keepout - self.agent_radius)*self.pixel_size,
                    width=round(self.pixel_size/10),
                )
            else:
                # Flag is captured so draw a different shape
                flag_pos_screen = self.world_to_screen(flag.pos)
                draw.circle(
                    screen,
                    color,
                    flag_pos_screen,
                    0.55*(self.pixel_size*self.agent_radius)
                )

            for player in teams_players:
                # render tagging
                player.render_tagging(self.tagging_cooldown)

                # heading
                orientation = Vector2(list(mag_heading_to_vec(1.0, player.heading)))
                ref_angle = -orientation.angle_to(self.UP)

                # transform position to pygame coordinates
                blit_pos = self.world_to_screen(player.pos)
                rotated_surface = rotozoom(player.pygame_agent, ref_angle, 1.0)
                rotated_surface_size = np.array(rotated_surface.get_size())
                rotated_blit_pos = blit_pos - 0.5*rotated_surface_size

                # blit agent onto screen
                screen.blit(rotated_surface, rotated_blit_pos)

                # blit agent number onto agent
                agent_id_blit_poses[player.id] = (
                    blit_pos[0] - 0.35*self.pixel_size*self.agent_radius,
                    blit_pos[1] - 0.6*self.pixel_size*self.agent_radius
                )

        # render agent ids
        if self.render_ids:
            for team in Team:
                teams_players = self.agents_of_team[team]
                font_color = "white" if team == Team.BLUE_TEAM else "black"
                for player in teams_players:
                    player_number_label = self.font.render(str(player.id), True, font_color)
                    screen.blit(player_number_label, agent_id_blit_poses[player.id])

        # visually indicate distances between players of both teams
        assert len(self.agents_of_team) == 2, "If number of teams > 2, update code that draws distance indicator lines"

        for blue_player in self.agents_of_team[Team.BLUE_TEAM]:
            if not blue_player.is_tagged or (blue_player.is_tagged and blue_player.on_own_side):
                for red_player in self.agents_of_team[Team.RED_TEAM]:
                    if not red_player.is_tagged or (red_player.is_tagged and red_player.on_own_side):
                        blue_player_pos = np.asarray(blue_player.pos)
                        red_player_pos = np.asarray(red_player.pos)
                        a2a_dis = np.linalg.norm(blue_player_pos - red_player_pos)
                        if a2a_dis <= 2*self.catch_radius:
                            hsv_hue = (a2a_dis - self.catch_radius)/(2*self.catch_radius - self.catch_radius)
                            hsv_hue = 0.33*np.clip(hsv_hue, 0, 1)
                            line_color = tuple(255*np.asarray(colorsys.hsv_to_rgb(hsv_hue, 0.9, 0.9)))

                            draw.line(
                                screen,
                                line_color,
                                self.world_to_screen(blue_player_pos),
                                self.world_to_screen(red_player_pos),
                                width=self.a2a_line_width
                            )

        return pygame.transform.flip(screen, False, True)


def policy_wrapper(Policy: BaseAgentPolicy, agent_obs_normalizer, identity='wrapped_policy'):
    class WrappedPolicy(Policy):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.identity = identity

        def set_team(self, team):
            if team == 0:
                self.team = Team.BLUE_TEAM
            else:
                self.team = Team.RED_TEAM

        def get_action(self, obs, *args, **kwargs):
            agent_obs = agent_obs_normalizer.unnormalized(obs)
            return self.compute_action({self.id: agent_obs})

    return WrappedPolicy
