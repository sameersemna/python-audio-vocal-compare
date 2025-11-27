from youtubesearchpython import ChannelSearch,ResultMode

search = ChannelSearch('Watermelon Sugar', "UCZFWPqqPkFlNwIxcpsLOwew")
print(search.result(mode = ResultMode.json))

# from youtubesearchpython import *

# channel_id = "UC_aEa8K-EOJ3D6gOs7HcyNg"
# playlist = Playlist(playlist_from_channel_id(channel_id))

# print(f'Videos Retrieved: {len(playlist.videos)}')

# while playlist.hasMoreVideos:
#     print('Getting more videos...')
#     playlist.getNextVideos()
#     print(f'Videos Retrieved: {len(playlist.videos)}')

# print('Found all the videos.')