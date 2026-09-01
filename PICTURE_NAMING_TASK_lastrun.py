#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
This experiment was created using PsychoPy3 Experiment Builder (v2025.2.0),
    on August 30, 2026, at 18:51
If you publish work using this script the most relevant publication is:

    Peirce J, Gray JR, Simpson S, MacAskill M, Höchenberger R, Sogo H, Kastman E, Lindeløv JK. (2019) 
        PsychoPy2: Experiments in behavior made easy Behav Res 51: 195. 
        https://doi.org/10.3758/s13428-018-01193-y

"""

# --- Import packages ---
from psychopy import locale_setup
from psychopy import prefs
from psychopy import plugins
plugins.activatePlugins()
prefs.hardware['audioLib'] = 'ptb'
from psychopy import sound, gui, visual, core, data, event, logging, clock, colors, layout, hardware
from psychopy.tools import environmenttools
from psychopy.constants import (
    NOT_STARTED, STARTED, PLAYING, PAUSED, STOPPED, STOPPING, FINISHED, PRESSED, 
    RELEASED, FOREVER, priority
)

import numpy as np  # whole numpy lib is available, prepend 'np.'
from numpy import (sin, cos, tan, log, log10, pi, average,
                   sqrt, std, deg2rad, rad2deg, linspace, asarray)
from numpy.random import random, randint, normal, shuffle, choice as randchoice
import os  # handy system and path functions
import sys  # to get file system encoding

from psychopy.hardware import keyboard

# --- Setup global variables (available in all functions) ---
# create a device manager to handle hardware (keyboards, mice, mirophones, speakers, etc.)
deviceManager = hardware.DeviceManager()
# ensure that relative paths start from the same directory as this script
_thisDir = os.path.dirname(os.path.abspath(__file__))
# store info about the experiment session
psychopyVersion = '2025.2.0'
expName = 'PICTURE_NAMING_TASK'  # from the Builder filename that created this script
expVersion = ''
# a list of functions to run when the experiment ends (starts off blank)
runAtExit = []
# information about this experiment
expInfo = {
    'participant': f"{randint(0, 999999):06.0f}",
    'session': '"001"',
    'date|hid': data.getDateStr(),
    'expName|hid': expName,
    'expVersion|hid': expVersion,
    'psychopyVersion|hid': psychopyVersion,
}

# --- Define some variables which will change depending on pilot mode ---
'''
To run in pilot mode, either use the run/pilot toggle in Builder, Coder and Runner, 
or run the experiment with `--pilot` as an argument. To change what pilot 
#mode does, check out the 'Pilot mode' tab in preferences.
'''
# work out from system args whether we are running in pilot mode
PILOTING = core.setPilotModeFromArgs()
# start off with values from experiment settings
_fullScr = True
_winSize = (1024, 768)
# if in pilot mode, apply overrides according to preferences
if PILOTING:
    # force windowed mode
    if prefs.piloting['forceWindowed']:
        _fullScr = False
        # set window size
        _winSize = prefs.piloting['forcedWindowSize']
    # replace default participant ID
    if prefs.piloting['replaceParticipantID']:
        expInfo['participant'] = 'pilot'

def showExpInfoDlg(expInfo):
    """
    Show participant info dialog.
    Parameters
    ==========
    expInfo : dict
        Information about this experiment.
    
    Returns
    ==========
    dict
        Information about this experiment.
    """
    # show participant info dialog
    dlg = gui.DlgFromDict(
        dictionary=expInfo, sortKeys=False, title=expName, alwaysOnTop=True
    )
    if dlg.OK == False:
        core.quit()  # user pressed cancel
    # return expInfo
    return expInfo


def setupData(expInfo, dataDir=None):
    """
    Make an ExperimentHandler to handle trials and saving.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    dataDir : Path, str or None
        Folder to save the data to, leave as None to create a folder in the current directory.    
    Returns
    ==========
    psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    # remove dialog-specific syntax from expInfo
    for key, val in expInfo.copy().items():
        newKey, _ = data.utils.parsePipeSyntax(key)
        expInfo[newKey] = expInfo.pop(key)
    
    # data file name stem = absolute path + name; later add .psyexp, .csv, .log, etc
    if dataDir is None:
        dataDir = _thisDir
    filename = u'data/%s_%s_%s' % (expInfo['participant'], expName, expInfo['date'])
    # make sure filename is relative to dataDir
    if os.path.isabs(filename):
        dataDir = os.path.commonprefix([dataDir, filename])
        filename = os.path.relpath(filename, dataDir)
    
    # an ExperimentHandler isn't essential but helps with data saving
    thisExp = data.ExperimentHandler(
        name=expName, version=expVersion,
        extraInfo=expInfo, runtimeInfo=None,
        originPath='D:\\COLLEGE\\🗀\\skripsi\\psychopy\\PICTURE_NAMING_TASK\\PICTURE_NAMING_TASK_lastrun.py',
        savePickle=True, saveWideText=True,
        dataFileName=dataDir + os.sep + filename, sortColumns='time'
    )
    thisExp.setPriority('thisRow.t', priority.CRITICAL)
    thisExp.setPriority('expName', priority.LOW)
    # return experiment handler
    return thisExp


def setupLogging(filename):
    """
    Setup a log file and tell it what level to log at.
    
    Parameters
    ==========
    filename : str or pathlib.Path
        Filename to save log file and data files as, doesn't need an extension.
    
    Returns
    ==========
    psychopy.logging.LogFile
        Text stream to receive inputs from the logging system.
    """
    # set how much information should be printed to the console / app
    if PILOTING:
        logging.console.setLevel(
            prefs.piloting['pilotConsoleLoggingLevel']
        )
    else:
        logging.console.setLevel('warning')
    # save a log file for detail verbose info
    logFile = logging.LogFile(filename+'.log')
    if PILOTING:
        logFile.setLevel(
            prefs.piloting['pilotLoggingLevel']
        )
    else:
        logFile.setLevel(
            logging.getLevel('info')
        )
    
    return logFile


def setupWindow(expInfo=None, win=None):
    """
    Setup the Window
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    win : psychopy.visual.Window
        Window to setup - leave as None to create a new window.
    
    Returns
    ==========
    psychopy.visual.Window
        Window in which to run this experiment.
    """
    if PILOTING:
        logging.debug('Fullscreen settings ignored as running in pilot mode.')
    
    if win is None:
        # if not given a window to setup, make one
        win = visual.Window(
            size=_winSize, fullscr=_fullScr, screen=0,
            winType='pyglet', allowGUI=False, allowStencil=False,
            monitor='testMonitor', color=[1,1,1], colorSpace='rgb',
            backgroundImage='', backgroundFit='none',
            blendMode='avg', useFBO=True,
            units='height',
            checkTiming=False  # we're going to do this ourselves in a moment
        )
    else:
        # if we have a window, just set the attributes which are safe to set
        win.color = [1,1,1]
        win.colorSpace = 'rgb'
        win.backgroundImage = ''
        win.backgroundFit = 'none'
        win.units = 'height'
    if expInfo is not None:
        # get/measure frame rate if not already in expInfo
        if win._monitorFrameRate is None:
            win._monitorFrameRate = win.getActualFrameRate(infoMsg='Attempting to measure frame rate of screen, please wait...')
        expInfo['frameRate'] = win._monitorFrameRate
    win.hideMessage()
    if PILOTING:
        # show a visual indicator if we're in piloting mode
        if prefs.piloting['showPilotingIndicator']:
            win.showPilotingIndicator()
        # always show the mouse in piloting mode
        if prefs.piloting['forceMouseVisible']:
            win.mouseVisible = True
    
    return win


def setupDevices(expInfo, thisExp, win):
    """
    Setup whatever devices are available (mouse, keyboard, speaker, eyetracker, etc.) and add them to 
    the device manager (deviceManager)
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window in which to run this experiment.
    Returns
    ==========
    bool
        True if completed successfully.
    """
    # --- Setup input devices ---
    ioConfig = {}
    ioSession = ioServer = eyetracker = None
    
    # store ioServer object in the device manager
    deviceManager.ioServer = ioServer
    
    # create a default keyboard (e.g. to check for escape)
    if deviceManager.getDevice('defaultKeyboard') is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='ptb'
        )
    # initialize 'usethis'
    deviceManager.addDevice(
        deviceName='usethis',
        deviceClass='psychopy.hardware.microphone.MicrophoneDevice',
        index='Microphone (Realtek(R) Audio)',
        sampleRateHz=48000.0,
        channels=2,
        exclusive=False,
    )
    # return True if completed successfully
    return True

def pauseExperiment(thisExp, win=None, timers=[], currentRoutine=None):
    """
    Pause this experiment, preventing the flow from advancing to the next routine until resumed.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    timers : list, tuple
        List of timers to reset once pausing is finished.
    currentRoutine : psychopy.data.Routine
        Current Routine we are in at time of pausing, if any. This object tells PsychoPy what Components to pause/play/dispatch.
    """
    # if we are not paused, do nothing
    if thisExp.status != PAUSED:
        return
    
    # start a timer to figure out how long we're paused for
    pauseTimer = core.Clock()
    # pause any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.pause()
    # make sure we have a keyboard
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        defaultKeyboard = deviceManager.addKeyboard(
            deviceClass='keyboard',
            deviceName='defaultKeyboard',
            backend='PsychToolbox',
        )
    # run a while loop while we wait to unpause
    while thisExp.status == PAUSED:
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=['escape']):
            endExperiment(thisExp, win=win)
        # dispatch messages on response components
        if currentRoutine is not None:
            for comp in currentRoutine.getDispatchComponents():
                comp.device.dispatchMessages()
        # sleep 1ms so other threads can execute
        clock.time.sleep(0.001)
    # if stop was requested while paused, quit
    if thisExp.status == FINISHED:
        endExperiment(thisExp, win=win)
    # resume any playback components
    if currentRoutine is not None:
        for comp in currentRoutine.getPlaybackComponents():
            comp.play()
    # reset any timers
    for timer in timers:
        timer.addTime(-pauseTimer.getTime())


def run(expInfo, thisExp, win, globalClock=None, thisSession=None):
    """
    Run the experiment flow.
    
    Parameters
    ==========
    expInfo : dict
        Information about this experiment, created by the `setupExpInfo` function.
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    psychopy.visual.Window
        Window in which to run this experiment.
    globalClock : psychopy.core.clock.Clock or None
        Clock to get global time from - supply None to make a new one.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    # mark experiment as started
    thisExp.status = STARTED
    # update experiment info
    expInfo['date'] = data.getDateStr()
    expInfo['expName'] = expName
    expInfo['expVersion'] = expVersion
    expInfo['psychopyVersion'] = psychopyVersion
    # make sure window is set to foreground to prevent losing focus
    win.winHandle.activate()
    # make sure variables created by exec are available globally
    exec = environmenttools.setExecEnvironment(globals())
    # get device handles from dict of input devices
    ioServer = deviceManager.ioServer
    # get/create a default keyboard (e.g. to check for escape)
    defaultKeyboard = deviceManager.getDevice('defaultKeyboard')
    if defaultKeyboard is None:
        deviceManager.addDevice(
            deviceClass='keyboard', deviceName='defaultKeyboard', backend='PsychToolbox'
        )
    eyetracker = deviceManager.getDevice('eyetracker')
    # make sure we're running in the directory for this experiment
    os.chdir(_thisDir)
    # get filename from ExperimentHandler for convenience
    filename = thisExp.dataFileName
    frameTolerance = 0.001  # how close to onset before 'same' frame
    endExpNow = False  # flag for 'escape' or other condition => quit the exp
    # get frame duration from frame rate in expInfo
    if 'frameRate' in expInfo and expInfo['frameRate'] is not None:
        frameDur = 1.0 / round(expInfo['frameRate'])
    else:
        frameDur = 1.0 / 60.0  # could not measure, so guess
    
    # Start Code - component code to be run after the window creation
    # Make folder to store recordings from practice_ID_response
    practice_ID_responseRecFolder = filename + '_practice_ID_response_recorded'
    if not os.path.isdir(practice_ID_responseRecFolder):
        os.mkdir(practice_ID_responseRecFolder)
    # Make folder to store recordings from practice_EN_response
    practice_EN_responseRecFolder = filename + '_practice_EN_response_recorded'
    if not os.path.isdir(practice_EN_responseRecFolder):
        os.mkdir(practice_EN_responseRecFolder)
    # Make folder to store recordings from practice_SWITCHING_response
    practice_SWITCHING_responseRecFolder = filename + '_practice_SWITCHING_response_recorded'
    if not os.path.isdir(practice_SWITCHING_responseRecFolder):
        os.mkdir(practice_SWITCHING_responseRecFolder)
    # Make folder to store recordings from ID_response
    ID_responseRecFolder = filename + '_ID_response_recorded'
    if not os.path.isdir(ID_responseRecFolder):
        os.mkdir(ID_responseRecFolder)
    # Make folder to store recordings from EN_response
    EN_responseRecFolder = filename + '_EN_response_recorded'
    if not os.path.isdir(EN_responseRecFolder):
        os.mkdir(EN_responseRecFolder)
    # Make folder to store recordings from SWITCHING_response
    SWITCHING_responseRecFolder = filename + '_SWITCHING_response_recorded'
    if not os.path.isdir(SWITCHING_responseRecFolder):
        os.mkdir(SWITCHING_responseRecFolder)
    
    # --- Initialize components for Routine "instruction" ---
    text_instruction = visual.TextStim(win=win, name='text_instruction',
        text='Anda akan melihat gambar pada setiap trial.\nSebutkan nama gambar tersebut secara lisan SEAKURAT mungkin.\n\nJika muncul bendera Indonesia, jawab dengan Bahasa Indonesia.\nJika muncul bendera Inggris, jawab dengan Bahasa Inggris.\n\nTekan spasi untuk melanjutkan.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "practice_block" ---
    text_practice_block = visual.TextStim(win=win, name='text_practice_block',
        text='Blok pertama adalah blok latihan.\nINGAT: jawab seakurat mungkin.\n\nTekan spasi untuk melanjutkan.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color='black', colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_2 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # --- Initialize components for Routine "practice_dot_ID" ---
    practice_fixationdot_IN = visual.ShapeStim(
        win=win, name='practice_fixationdot_IN',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "practice_picstim_ID" ---
    practice_languagecue_ID = visual.ImageStim(
        win=win,
        name='practice_languagecue_ID', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    practice_picturestimuli_ID = visual.ImageStim(
        win=win,
        name='practice_picturestimuli_ID', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # make microphone object for practice_ID_response
    practice_ID_response = sound.microphone.Microphone(
        device='usethis',
        name='practice_ID_response',
        recordingFolder=practice_ID_responseRecFolder,
        recordingExt='wav'
    )
    # tell the experiment handler to save this Microphone's clips if the experiment is force ended
    runAtExit.append(practice_ID_response.saveClips)
    # connect camera save method to experiment handler so it's called when data saves
    thisExp.connectSaveMethod(practice_ID_response.saveClips)
    
    # --- Initialize components for Routine "practice_blank_ID" ---
    practice_blankscreen_IN = visual.ImageStim(
        win=win,
        name='practice_blankscreen_IN', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "practice_dot_EN" ---
    practice_fixationdot_EN = visual.ShapeStim(
        win=win, name='practice_fixationdot_EN',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "practice_picstim_EN" ---
    practice_languagecue_EN = visual.ImageStim(
        win=win,
        name='practice_languagecue_EN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    practice_picturestimuli_EN = visual.ImageStim(
        win=win,
        name='practice_picturestimuli_EN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # make microphone object for practice_EN_response
    practice_EN_response = sound.microphone.Microphone(
        device='usethis',
        name='practice_EN_response',
        recordingFolder=practice_EN_responseRecFolder,
        recordingExt='wav'
    )
    # tell the experiment handler to save this Microphone's clips if the experiment is force ended
    runAtExit.append(practice_EN_response.saveClips)
    # connect camera save method to experiment handler so it's called when data saves
    thisExp.connectSaveMethod(practice_EN_response.saveClips)
    
    # --- Initialize components for Routine "practice_blank_EN" ---
    practice_blankscreen_EN = visual.ImageStim(
        win=win,
        name='practice_blankscreen_EN', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "practice_dot_SWITCHING" ---
    practice_fixationdot_SWITCH = visual.ShapeStim(
        win=win, name='practice_fixationdot_SWITCH',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "practice_picstim_SWITCHING" ---
    practice_languagecue_SWITCHING = visual.ImageStim(
        win=win,
        name='practice_languagecue_SWITCHING', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    practice_picturestimuli_SWITCHING = visual.ImageStim(
        win=win,
        name='practice_picturestimuli_SWITCHING', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # make microphone object for practice_SWITCHING_response
    practice_SWITCHING_response = sound.microphone.Microphone(
        device='usethis',
        name='practice_SWITCHING_response',
        recordingFolder=practice_SWITCHING_responseRecFolder,
        recordingExt='wav'
    )
    # tell the experiment handler to save this Microphone's clips if the experiment is force ended
    runAtExit.append(practice_SWITCHING_response.saveClips)
    # connect camera save method to experiment handler so it's called when data saves
    thisExp.connectSaveMethod(practice_SWITCHING_response.saveClips)
    
    # --- Initialize components for Routine "practice_blank_SWITCHING" ---
    practice_blankscreen_SWITCHING = visual.ImageStim(
        win=win,
        name='practice_blankscreen_SWITCHING', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "nonswitching_block" ---
    text_nonswitching_block = visual.TextStim(win=win, name='text_nonswitching_block',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "dot_ID" ---
    fixationdot_ID = visual.ShapeStim(
        win=win, name='fixationdot_ID',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "picstim_ID" ---
    languagecue_ID = visual.ImageStim(
        win=win,
        name='languagecue_ID', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    picturestimuli_ID = visual.ImageStim(
        win=win,
        name='picturestimuli_ID', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # make microphone object for ID_response
    ID_response = sound.microphone.Microphone(
        device='usethis',
        name='ID_response',
        recordingFolder=ID_responseRecFolder,
        recordingExt='wav'
    )
    # tell the experiment handler to save this Microphone's clips if the experiment is force ended
    runAtExit.append(ID_response.saveClips)
    # connect camera save method to experiment handler so it's called when data saves
    thisExp.connectSaveMethod(ID_response.saveClips)
    
    # --- Initialize components for Routine "blank_ID" ---
    blankscreen_ID = visual.ImageStim(
        win=win,
        name='blankscreen_ID', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "dot_EN" ---
    fixationdot_EN = visual.ShapeStim(
        win=win, name='fixationdot_EN',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "picstim_EN" ---
    languagecue_EN = visual.ImageStim(
        win=win,
        name='languagecue_EN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    picturestimuli_EN = visual.ImageStim(
        win=win,
        name='picturestimuli_EN', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # make microphone object for EN_response
    EN_response = sound.microphone.Microphone(
        device='usethis',
        name='EN_response',
        recordingFolder=EN_responseRecFolder,
        recordingExt='wav'
    )
    # tell the experiment handler to save this Microphone's clips if the experiment is force ended
    runAtExit.append(EN_response.saveClips)
    # connect camera save method to experiment handler so it's called when data saves
    thisExp.connectSaveMethod(EN_response.saveClips)
    
    # --- Initialize components for Routine "blank_EN" ---
    blankscreen_EN = visual.ImageStim(
        win=win,
        name='blankscreen_EN', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "switching_block" ---
    text_switching_block = visual.TextStim(win=win, name='text_switching_block',
        text=None,
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    
    # --- Initialize components for Routine "dot_SWITCHING" ---
    fixationdot_MIX = visual.ShapeStim(
        win=win, name='fixationdot_MIX',
        size=(0.05, 0.05), vertices='circle',
        ori=0.0, pos=(0, 0), draggable=False, anchor='center',
        lineWidth=1.0,
        colorSpace='rgb', lineColor=(0.0000, 0.0000, 0.0000), fillColor='black',
        opacity=None, depth=0.0, interpolate=True)
    
    # --- Initialize components for Routine "picstim_SWITCHING" ---
    languagecue_SWITCHING = visual.ImageStim(
        win=win,
        name='languagecue_SWITCHING', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, 0.3), draggable=False, size=(0.15, 0.1),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    picturestimuli_SWITCHING = visual.ImageStim(
        win=win,
        name='picturestimuli_SWITCHING', 
        image='default.png', mask=None, anchor='center',
        ori=0.0, pos=(0, -0.04), draggable=False, size=(0.6, 0.6),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=-1.0)
    # make microphone object for SWITCHING_response
    SWITCHING_response = sound.microphone.Microphone(
        device='usethis',
        name='SWITCHING_response',
        recordingFolder=SWITCHING_responseRecFolder,
        recordingExt='wav'
    )
    # tell the experiment handler to save this Microphone's clips if the experiment is force ended
    runAtExit.append(SWITCHING_response.saveClips)
    # connect camera save method to experiment handler so it's called when data saves
    thisExp.connectSaveMethod(SWITCHING_response.saveClips)
    
    # --- Initialize components for Routine "blank_SWITCHING" ---
    blankscreen_SWITCH = visual.ImageStim(
        win=win,
        name='blankscreen_SWITCH', 
        image=None, mask=None, anchor='center',
        ori=0.0, pos=(0, 0), draggable=False, size=(0.5, 0.5),
        color=[1,1,1], colorSpace='rgb', opacity=None,
        flipHoriz=False, flipVert=False,
        texRes=128.0, interpolate=True, depth=0.0)
    
    # --- Initialize components for Routine "experiment_end" ---
    text = visual.TextStim(win=win, name='text',
        text='Selamat, Anda telah menyelesaikan tugas ini!\nTerima kasih atas partisipasi Anda.\n\nTekan spasi untuk keluar.',
        font='Arial',
        pos=(0, 0), draggable=False, height=0.04, wrapWidth=None, ori=0.0, 
        color=(-1.0000, -1.0000, -1.0000), colorSpace='rgb', opacity=None, 
        languageStyle='LTR',
        depth=0.0);
    key_resp_3 = keyboard.Keyboard(deviceName='defaultKeyboard')
    
    # create some handy timers
    
    # global clock to track the time since experiment started
    if globalClock is None:
        # create a clock if not given one
        globalClock = core.Clock()
    if isinstance(globalClock, str):
        # if given a string, make a clock accoridng to it
        if globalClock == 'float':
            # get timestamps as a simple value
            globalClock = core.Clock(format='float')
        elif globalClock == 'iso':
            # get timestamps in ISO format
            globalClock = core.Clock(format='%Y-%m-%d_%H:%M:%S.%f%z')
        else:
            # get timestamps in a custom format
            globalClock = core.Clock(format=globalClock)
    if ioServer is not None:
        ioServer.syncClock(globalClock)
    logging.setDefaultClock(globalClock)
    # routine timer to track time remaining of each (possibly non-slip) routine
    routineTimer = core.Clock()
    win.flip()  # flip window to reset last flip timer
    # store the exact time the global clock started
    expInfo['expStart'] = data.getDateStr(
        format='%Y-%m-%d %Hh%M.%S.%f %z', fractionalSecondDigits=6
    )
    
    # --- Prepare to start Routine "instruction" ---
    # create an object to store info about Routine instruction
    instruction = data.Routine(
        name='instruction',
        components=[text_instruction, key_resp],
    )
    instruction.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp
    key_resp.keys = []
    key_resp.rt = []
    _key_resp_allKeys = []
    # store start times for instruction
    instruction.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    instruction.tStart = globalClock.getTime(format='float')
    instruction.status = STARTED
    thisExp.addData('instruction.started', instruction.tStart)
    instruction.maxDuration = None
    # keep track of which components have finished
    instructionComponents = instruction.components
    for thisComponent in instruction.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "instruction" ---
    thisExp.currentRoutine = instruction
    instruction.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_instruction* updates
        
        # if text_instruction is starting this frame...
        if text_instruction.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_instruction.frameNStart = frameN  # exact frame index
            text_instruction.tStart = t  # local t and not account for scr refresh
            text_instruction.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_instruction, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_instruction.started')
            # update status
            text_instruction.status = STARTED
            text_instruction.setAutoDraw(True)
        
        # if text_instruction is active this frame...
        if text_instruction.status == STARTED:
            # update params
            pass
        
        # *key_resp* updates
        waitOnFlip = False
        
        # if key_resp is starting this frame...
        if key_resp.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp.frameNStart = frameN  # exact frame index
            key_resp.tStart = t  # local t and not account for scr refresh
            key_resp.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp.started')
            # update status
            key_resp.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp.status == STARTED and not waitOnFlip:
            theseKeys = key_resp.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_allKeys.extend(theseKeys)
            if len(_key_resp_allKeys):
                key_resp.keys = _key_resp_allKeys[-1].name  # just the last key pressed
                key_resp.rt = _key_resp_allKeys[-1].rt
                key_resp.duration = _key_resp_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=instruction,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            instruction.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if instruction.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in instruction.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "instruction" ---
    for thisComponent in instruction.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for instruction
    instruction.tStop = globalClock.getTime(format='float')
    instruction.tStopRefresh = tThisFlipGlobal
    thisExp.addData('instruction.stopped', instruction.tStop)
    # check responses
    if key_resp.keys in ['', [], None]:  # No response was made
        key_resp.keys = None
    thisExp.addData('key_resp.keys',key_resp.keys)
    if key_resp.keys != None:  # we had a response
        thisExp.addData('key_resp.rt', key_resp.rt)
        thisExp.addData('key_resp.duration', key_resp.duration)
    thisExp.nextEntry()
    # the Routine "instruction" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # --- Prepare to start Routine "practice_block" ---
    # create an object to store info about Routine practice_block
    practice_block = data.Routine(
        name='practice_block',
        components=[text_practice_block, key_resp_2],
    )
    practice_block.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_2
    key_resp_2.keys = []
    key_resp_2.rt = []
    _key_resp_2_allKeys = []
    # store start times for practice_block
    practice_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    practice_block.tStart = globalClock.getTime(format='float')
    practice_block.status = STARTED
    thisExp.addData('practice_block.started', practice_block.tStart)
    practice_block.maxDuration = None
    # keep track of which components have finished
    practice_blockComponents = practice_block.components
    for thisComponent in practice_block.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "practice_block" ---
    thisExp.currentRoutine = practice_block
    practice_block.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_practice_block* updates
        
        # if text_practice_block is starting this frame...
        if text_practice_block.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_practice_block.frameNStart = frameN  # exact frame index
            text_practice_block.tStart = t  # local t and not account for scr refresh
            text_practice_block.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_practice_block, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_practice_block.started')
            # update status
            text_practice_block.status = STARTED
            text_practice_block.setAutoDraw(True)
        
        # if text_practice_block is active this frame...
        if text_practice_block.status == STARTED:
            # update params
            pass
        
        # *key_resp_2* updates
        waitOnFlip = False
        
        # if key_resp_2 is starting this frame...
        if key_resp_2.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_2.frameNStart = frameN  # exact frame index
            key_resp_2.tStart = t  # local t and not account for scr refresh
            key_resp_2.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_2, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_2.started')
            # update status
            key_resp_2.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_2.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_2.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_2.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_2.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_2_allKeys.extend(theseKeys)
            if len(_key_resp_2_allKeys):
                key_resp_2.keys = _key_resp_2_allKeys[-1].name  # just the last key pressed
                key_resp_2.rt = _key_resp_2_allKeys[-1].rt
                key_resp_2.duration = _key_resp_2_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=practice_block,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            practice_block.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if practice_block.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in practice_block.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "practice_block" ---
    for thisComponent in practice_block.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for practice_block
    practice_block.tStop = globalClock.getTime(format='float')
    practice_block.tStopRefresh = tThisFlipGlobal
    thisExp.addData('practice_block.stopped', practice_block.tStop)
    # check responses
    if key_resp_2.keys in ['', [], None]:  # No response was made
        key_resp_2.keys = None
    thisExp.addData('key_resp_2.keys',key_resp_2.keys)
    if key_resp_2.keys != None:  # we had a response
        thisExp.addData('key_resp_2.rt', key_resp_2.rt)
        thisExp.addData('key_resp_2.duration', key_resp_2.duration)
    thisExp.nextEntry()
    # the Routine "practice_block" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    practice_ID_block = data.TrialHandler2(
        name='practice_ID_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition/condition_practice_ID.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(practice_ID_block)  # add the loop to the experiment
    thisPractice_ID_block = practice_ID_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_ID_block.rgb)
    if thisPractice_ID_block != None:
        for paramName in thisPractice_ID_block:
            globals()[paramName] = thisPractice_ID_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_ID_block in practice_ID_block:
        practice_ID_block.status = STARTED
        if hasattr(thisPractice_ID_block, 'status'):
            thisPractice_ID_block.status = STARTED
        currentLoop = practice_ID_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_ID_block.rgb)
        if thisPractice_ID_block != None:
            for paramName in thisPractice_ID_block:
                globals()[paramName] = thisPractice_ID_block[paramName]
        
        # --- Prepare to start Routine "practice_dot_ID" ---
        # create an object to store info about Routine practice_dot_ID
        practice_dot_ID = data.Routine(
            name='practice_dot_ID',
            components=[practice_fixationdot_IN],
        )
        practice_dot_ID.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for practice_dot_ID
        practice_dot_ID.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_dot_ID.tStart = globalClock.getTime(format='float')
        practice_dot_ID.status = STARTED
        thisExp.addData('practice_dot_ID.started', practice_dot_ID.tStart)
        practice_dot_ID.maxDuration = None
        # keep track of which components have finished
        practice_dot_IDComponents = practice_dot_ID.components
        for thisComponent in practice_dot_ID.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_dot_ID" ---
        thisExp.currentRoutine = practice_dot_ID
        practice_dot_ID.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_ID_block, 'status') and thisPractice_ID_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_fixationdot_IN* updates
            
            # if practice_fixationdot_IN is starting this frame...
            if practice_fixationdot_IN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_fixationdot_IN.frameNStart = frameN  # exact frame index
                practice_fixationdot_IN.tStart = t  # local t and not account for scr refresh
                practice_fixationdot_IN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_fixationdot_IN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_fixationdot_IN.started')
                # update status
                practice_fixationdot_IN.status = STARTED
                practice_fixationdot_IN.setAutoDraw(True)
            
            # if practice_fixationdot_IN is active this frame...
            if practice_fixationdot_IN.status == STARTED:
                # update params
                pass
            
            # if practice_fixationdot_IN is stopping this frame...
            if practice_fixationdot_IN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_fixationdot_IN.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_fixationdot_IN.tStop = t  # not accounting for scr refresh
                    practice_fixationdot_IN.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_fixationdot_IN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_fixationdot_IN.stopped')
                    # update status
                    practice_fixationdot_IN.status = FINISHED
                    practice_fixationdot_IN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_dot_ID,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_dot_ID.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_dot_ID.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_dot_ID.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_dot_ID" ---
        for thisComponent in practice_dot_ID.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_dot_ID
        practice_dot_ID.tStop = globalClock.getTime(format='float')
        practice_dot_ID.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_dot_ID.stopped', practice_dot_ID.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_dot_ID.maxDurationReached:
            routineTimer.addTime(-practice_dot_ID.maxDuration)
        elif practice_dot_ID.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "practice_picstim_ID" ---
        # create an object to store info about Routine practice_picstim_ID
        practice_picstim_ID = data.Routine(
            name='practice_picstim_ID',
            components=[practice_languagecue_ID, practice_picturestimuli_ID, practice_ID_response],
        )
        practice_picstim_ID.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        practice_languagecue_ID.setImage(path_lc_practice_ID)
        practice_picturestimuli_ID.setImage(path_ps_practice_ID)
        practice_ID_response.setPolicyWhenFull('warn')
        # store start times for practice_picstim_ID
        practice_picstim_ID.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_picstim_ID.tStart = globalClock.getTime(format='float')
        practice_picstim_ID.status = STARTED
        thisExp.addData('practice_picstim_ID.started', practice_picstim_ID.tStart)
        practice_picstim_ID.maxDuration = None
        # keep track of which components have finished
        practice_picstim_IDComponents = practice_picstim_ID.components
        for thisComponent in practice_picstim_ID.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_picstim_ID" ---
        thisExp.currentRoutine = practice_picstim_ID
        practice_picstim_ID.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_ID_block, 'status') and thisPractice_ID_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_languagecue_ID* updates
            
            # if practice_languagecue_ID is starting this frame...
            if practice_languagecue_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_languagecue_ID.frameNStart = frameN  # exact frame index
                practice_languagecue_ID.tStart = t  # local t and not account for scr refresh
                practice_languagecue_ID.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_languagecue_ID, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_languagecue_ID.started')
                # update status
                practice_languagecue_ID.status = STARTED
                practice_languagecue_ID.setAutoDraw(True)
            
            # if practice_languagecue_ID is active this frame...
            if practice_languagecue_ID.status == STARTED:
                # update params
                pass
            
            # if practice_languagecue_ID is stopping this frame...
            if practice_languagecue_ID.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_languagecue_ID.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_languagecue_ID.tStop = t  # not accounting for scr refresh
                    practice_languagecue_ID.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_languagecue_ID.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_languagecue_ID.stopped')
                    # update status
                    practice_languagecue_ID.status = FINISHED
                    practice_languagecue_ID.setAutoDraw(False)
            
            # *practice_picturestimuli_ID* updates
            
            # if practice_picturestimuli_ID is starting this frame...
            if practice_picturestimuli_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_picturestimuli_ID.frameNStart = frameN  # exact frame index
                practice_picturestimuli_ID.tStart = t  # local t and not account for scr refresh
                practice_picturestimuli_ID.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_picturestimuli_ID, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_picturestimuli_ID.started')
                # update status
                practice_picturestimuli_ID.status = STARTED
                practice_picturestimuli_ID.setAutoDraw(True)
            
            # if practice_picturestimuli_ID is active this frame...
            if practice_picturestimuli_ID.status == STARTED:
                # update params
                pass
            
            # if practice_picturestimuli_ID is stopping this frame...
            if practice_picturestimuli_ID.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_picturestimuli_ID.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_picturestimuli_ID.tStop = t  # not accounting for scr refresh
                    practice_picturestimuli_ID.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_picturestimuli_ID.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_picturestimuli_ID.stopped')
                    # update status
                    practice_picturestimuli_ID.status = FINISHED
                    practice_picturestimuli_ID.setAutoDraw(False)
            
            # if practice_ID_response is starting this frame...
            if practice_ID_response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_ID_response.frameNStart = frameN  # exact frame index
                practice_ID_response.tStart = t  # local t and not account for scr refresh
                practice_ID_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_ID_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('practice_ID_response.started', t)
                # update status
                practice_ID_response.status = STARTED
                # start recording with practice_ID_response
                practice_ID_response.start()
            
            # if practice_ID_response is active this frame...
            if practice_ID_response.status == STARTED:
                # update params
                pass
                # update recorded clip for practice_ID_response
                practice_ID_response.poll()
            
            # if practice_ID_response is stopping this frame...
            if practice_ID_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_ID_response.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_ID_response.tStop = t  # not accounting for scr refresh
                    practice_ID_response.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_ID_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('practice_ID_response.stopped', t)
                    # update status
                    practice_ID_response.status = FINISHED
                    # stop recording with practice_ID_response
                    practice_ID_response.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_picstim_ID,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_picstim_ID.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_picstim_ID.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_picstim_ID.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_picstim_ID" ---
        for thisComponent in practice_picstim_ID.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_picstim_ID
        practice_picstim_ID.tStop = globalClock.getTime(format='float')
        practice_picstim_ID.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_picstim_ID.stopped', practice_picstim_ID.tStop)
        # tell mic to keep hold of current recording in practice_ID_response.clips and transcript (if applicable) in practice_ID_response.scripts
        # this will also update practice_ID_response.lastClip and practice_ID_response.lastScript
        practice_ID_response.stop()
        tag = data.utils.getDateStr()
        practice_ID_responseClip = practice_ID_response.bank(
            tag=tag, transcribe='None',
            config=None
        )
        practice_ID_block.addData(
            'practice_ID_response.clip', practice_ID_response.recordingFolder / practice_ID_response.getClipFilename(tag)
        )
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_picstim_ID.maxDurationReached:
            routineTimer.addTime(-practice_picstim_ID.maxDuration)
        elif practice_picstim_ID.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "practice_blank_ID" ---
        # create an object to store info about Routine practice_blank_ID
        practice_blank_ID = data.Routine(
            name='practice_blank_ID',
            components=[practice_blankscreen_IN],
        )
        practice_blank_ID.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for practice_blank_ID
        practice_blank_ID.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_blank_ID.tStart = globalClock.getTime(format='float')
        practice_blank_ID.status = STARTED
        thisExp.addData('practice_blank_ID.started', practice_blank_ID.tStart)
        practice_blank_ID.maxDuration = None
        # keep track of which components have finished
        practice_blank_IDComponents = practice_blank_ID.components
        for thisComponent in practice_blank_ID.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_blank_ID" ---
        thisExp.currentRoutine = practice_blank_ID
        practice_blank_ID.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_ID_block, 'status') and thisPractice_ID_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_blankscreen_IN* updates
            
            # if practice_blankscreen_IN is starting this frame...
            if practice_blankscreen_IN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_blankscreen_IN.frameNStart = frameN  # exact frame index
                practice_blankscreen_IN.tStart = t  # local t and not account for scr refresh
                practice_blankscreen_IN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_blankscreen_IN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_blankscreen_IN.started')
                # update status
                practice_blankscreen_IN.status = STARTED
                practice_blankscreen_IN.setAutoDraw(True)
            
            # if practice_blankscreen_IN is active this frame...
            if practice_blankscreen_IN.status == STARTED:
                # update params
                pass
            
            # if practice_blankscreen_IN is stopping this frame...
            if practice_blankscreen_IN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_blankscreen_IN.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_blankscreen_IN.tStop = t  # not accounting for scr refresh
                    practice_blankscreen_IN.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_blankscreen_IN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_blankscreen_IN.stopped')
                    # update status
                    practice_blankscreen_IN.status = FINISHED
                    practice_blankscreen_IN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_blank_ID,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_blank_ID.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_blank_ID.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_blank_ID.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_blank_ID" ---
        for thisComponent in practice_blank_ID.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_blank_ID
        practice_blank_ID.tStop = globalClock.getTime(format='float')
        practice_blank_ID.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_blank_ID.stopped', practice_blank_ID.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_blank_ID.maxDurationReached:
            routineTimer.addTime(-practice_blank_ID.maxDuration)
        elif practice_blank_ID.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPractice_ID_block as finished
        if hasattr(thisPractice_ID_block, 'status'):
            thisPractice_ID_block.status = FINISHED
        # if awaiting a pause, pause now
        if practice_ID_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practice_ID_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_ID_block'
    practice_ID_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if practice_ID_block.trialList in ([], [None], None):
        params = []
    else:
        params = practice_ID_block.trialList[0].keys()
    # save data for this loop
    practice_ID_block.saveAsExcel(filename + '.xlsx', sheetName='practice_ID_block',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # set up handler to look after randomisation of conditions etc
    practice_EN_block = data.TrialHandler2(
        name='practice_EN_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition/condition_practice_EN.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(practice_EN_block)  # add the loop to the experiment
    thisPractice_EN_block = practice_EN_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_EN_block.rgb)
    if thisPractice_EN_block != None:
        for paramName in thisPractice_EN_block:
            globals()[paramName] = thisPractice_EN_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_EN_block in practice_EN_block:
        practice_EN_block.status = STARTED
        if hasattr(thisPractice_EN_block, 'status'):
            thisPractice_EN_block.status = STARTED
        currentLoop = practice_EN_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_EN_block.rgb)
        if thisPractice_EN_block != None:
            for paramName in thisPractice_EN_block:
                globals()[paramName] = thisPractice_EN_block[paramName]
        
        # --- Prepare to start Routine "practice_dot_EN" ---
        # create an object to store info about Routine practice_dot_EN
        practice_dot_EN = data.Routine(
            name='practice_dot_EN',
            components=[practice_fixationdot_EN],
        )
        practice_dot_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for practice_dot_EN
        practice_dot_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_dot_EN.tStart = globalClock.getTime(format='float')
        practice_dot_EN.status = STARTED
        thisExp.addData('practice_dot_EN.started', practice_dot_EN.tStart)
        practice_dot_EN.maxDuration = None
        # keep track of which components have finished
        practice_dot_ENComponents = practice_dot_EN.components
        for thisComponent in practice_dot_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_dot_EN" ---
        thisExp.currentRoutine = practice_dot_EN
        practice_dot_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_EN_block, 'status') and thisPractice_EN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_fixationdot_EN* updates
            
            # if practice_fixationdot_EN is starting this frame...
            if practice_fixationdot_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_fixationdot_EN.frameNStart = frameN  # exact frame index
                practice_fixationdot_EN.tStart = t  # local t and not account for scr refresh
                practice_fixationdot_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_fixationdot_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_fixationdot_EN.started')
                # update status
                practice_fixationdot_EN.status = STARTED
                practice_fixationdot_EN.setAutoDraw(True)
            
            # if practice_fixationdot_EN is active this frame...
            if practice_fixationdot_EN.status == STARTED:
                # update params
                pass
            
            # if practice_fixationdot_EN is stopping this frame...
            if practice_fixationdot_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_fixationdot_EN.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_fixationdot_EN.tStop = t  # not accounting for scr refresh
                    practice_fixationdot_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_fixationdot_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_fixationdot_EN.stopped')
                    # update status
                    practice_fixationdot_EN.status = FINISHED
                    practice_fixationdot_EN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_dot_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_dot_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_dot_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_dot_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_dot_EN" ---
        for thisComponent in practice_dot_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_dot_EN
        practice_dot_EN.tStop = globalClock.getTime(format='float')
        practice_dot_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_dot_EN.stopped', practice_dot_EN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_dot_EN.maxDurationReached:
            routineTimer.addTime(-practice_dot_EN.maxDuration)
        elif practice_dot_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "practice_picstim_EN" ---
        # create an object to store info about Routine practice_picstim_EN
        practice_picstim_EN = data.Routine(
            name='practice_picstim_EN',
            components=[practice_languagecue_EN, practice_picturestimuli_EN, practice_EN_response],
        )
        practice_picstim_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        practice_languagecue_EN.setImage(path_lc_practice_EN)
        practice_picturestimuli_EN.setImage(path_ps_practice_EN)
        practice_EN_response.setPolicyWhenFull('warn')
        # store start times for practice_picstim_EN
        practice_picstim_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_picstim_EN.tStart = globalClock.getTime(format='float')
        practice_picstim_EN.status = STARTED
        thisExp.addData('practice_picstim_EN.started', practice_picstim_EN.tStart)
        practice_picstim_EN.maxDuration = None
        # keep track of which components have finished
        practice_picstim_ENComponents = practice_picstim_EN.components
        for thisComponent in practice_picstim_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_picstim_EN" ---
        thisExp.currentRoutine = practice_picstim_EN
        practice_picstim_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_EN_block, 'status') and thisPractice_EN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_languagecue_EN* updates
            
            # if practice_languagecue_EN is starting this frame...
            if practice_languagecue_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_languagecue_EN.frameNStart = frameN  # exact frame index
                practice_languagecue_EN.tStart = t  # local t and not account for scr refresh
                practice_languagecue_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_languagecue_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_languagecue_EN.started')
                # update status
                practice_languagecue_EN.status = STARTED
                practice_languagecue_EN.setAutoDraw(True)
            
            # if practice_languagecue_EN is active this frame...
            if practice_languagecue_EN.status == STARTED:
                # update params
                pass
            
            # if practice_languagecue_EN is stopping this frame...
            if practice_languagecue_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_languagecue_EN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_languagecue_EN.tStop = t  # not accounting for scr refresh
                    practice_languagecue_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_languagecue_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_languagecue_EN.stopped')
                    # update status
                    practice_languagecue_EN.status = FINISHED
                    practice_languagecue_EN.setAutoDraw(False)
            
            # *practice_picturestimuli_EN* updates
            
            # if practice_picturestimuli_EN is starting this frame...
            if practice_picturestimuli_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_picturestimuli_EN.frameNStart = frameN  # exact frame index
                practice_picturestimuli_EN.tStart = t  # local t and not account for scr refresh
                practice_picturestimuli_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_picturestimuli_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_picturestimuli_EN.started')
                # update status
                practice_picturestimuli_EN.status = STARTED
                practice_picturestimuli_EN.setAutoDraw(True)
            
            # if practice_picturestimuli_EN is active this frame...
            if practice_picturestimuli_EN.status == STARTED:
                # update params
                pass
            
            # if practice_picturestimuli_EN is stopping this frame...
            if practice_picturestimuli_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_picturestimuli_EN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_picturestimuli_EN.tStop = t  # not accounting for scr refresh
                    practice_picturestimuli_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_picturestimuli_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_picturestimuli_EN.stopped')
                    # update status
                    practice_picturestimuli_EN.status = FINISHED
                    practice_picturestimuli_EN.setAutoDraw(False)
            
            # if practice_EN_response is starting this frame...
            if practice_EN_response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_EN_response.frameNStart = frameN  # exact frame index
                practice_EN_response.tStart = t  # local t and not account for scr refresh
                practice_EN_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_EN_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('practice_EN_response.started', t)
                # update status
                practice_EN_response.status = STARTED
                # start recording with practice_EN_response
                practice_EN_response.start()
            
            # if practice_EN_response is active this frame...
            if practice_EN_response.status == STARTED:
                # update params
                pass
                # update recorded clip for practice_EN_response
                practice_EN_response.poll()
            
            # if practice_EN_response is stopping this frame...
            if practice_EN_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_EN_response.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_EN_response.tStop = t  # not accounting for scr refresh
                    practice_EN_response.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_EN_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('practice_EN_response.stopped', t)
                    # update status
                    practice_EN_response.status = FINISHED
                    # stop recording with practice_EN_response
                    practice_EN_response.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_picstim_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_picstim_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_picstim_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_picstim_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_picstim_EN" ---
        for thisComponent in practice_picstim_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_picstim_EN
        practice_picstim_EN.tStop = globalClock.getTime(format='float')
        practice_picstim_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_picstim_EN.stopped', practice_picstim_EN.tStop)
        # tell mic to keep hold of current recording in practice_EN_response.clips and transcript (if applicable) in practice_EN_response.scripts
        # this will also update practice_EN_response.lastClip and practice_EN_response.lastScript
        practice_EN_response.stop()
        tag = data.utils.getDateStr()
        practice_EN_responseClip = practice_EN_response.bank(
            tag=tag, transcribe='None',
            config=None
        )
        practice_EN_block.addData(
            'practice_EN_response.clip', practice_EN_response.recordingFolder / practice_EN_response.getClipFilename(tag)
        )
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_picstim_EN.maxDurationReached:
            routineTimer.addTime(-practice_picstim_EN.maxDuration)
        elif practice_picstim_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "practice_blank_EN" ---
        # create an object to store info about Routine practice_blank_EN
        practice_blank_EN = data.Routine(
            name='practice_blank_EN',
            components=[practice_blankscreen_EN],
        )
        practice_blank_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for practice_blank_EN
        practice_blank_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_blank_EN.tStart = globalClock.getTime(format='float')
        practice_blank_EN.status = STARTED
        thisExp.addData('practice_blank_EN.started', practice_blank_EN.tStart)
        practice_blank_EN.maxDuration = None
        # keep track of which components have finished
        practice_blank_ENComponents = practice_blank_EN.components
        for thisComponent in practice_blank_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_blank_EN" ---
        thisExp.currentRoutine = practice_blank_EN
        practice_blank_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_EN_block, 'status') and thisPractice_EN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_blankscreen_EN* updates
            
            # if practice_blankscreen_EN is starting this frame...
            if practice_blankscreen_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_blankscreen_EN.frameNStart = frameN  # exact frame index
                practice_blankscreen_EN.tStart = t  # local t and not account for scr refresh
                practice_blankscreen_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_blankscreen_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_blankscreen_EN.started')
                # update status
                practice_blankscreen_EN.status = STARTED
                practice_blankscreen_EN.setAutoDraw(True)
            
            # if practice_blankscreen_EN is active this frame...
            if practice_blankscreen_EN.status == STARTED:
                # update params
                pass
            
            # if practice_blankscreen_EN is stopping this frame...
            if practice_blankscreen_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_blankscreen_EN.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_blankscreen_EN.tStop = t  # not accounting for scr refresh
                    practice_blankscreen_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_blankscreen_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_blankscreen_EN.stopped')
                    # update status
                    practice_blankscreen_EN.status = FINISHED
                    practice_blankscreen_EN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_blank_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_blank_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_blank_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_blank_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_blank_EN" ---
        for thisComponent in practice_blank_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_blank_EN
        practice_blank_EN.tStop = globalClock.getTime(format='float')
        practice_blank_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_blank_EN.stopped', practice_blank_EN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_blank_EN.maxDurationReached:
            routineTimer.addTime(-practice_blank_EN.maxDuration)
        elif practice_blank_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPractice_EN_block as finished
        if hasattr(thisPractice_EN_block, 'status'):
            thisPractice_EN_block.status = FINISHED
        # if awaiting a pause, pause now
        if practice_EN_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practice_EN_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_EN_block'
    practice_EN_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if practice_EN_block.trialList in ([], [None], None):
        params = []
    else:
        params = practice_EN_block.trialList[0].keys()
    # save data for this loop
    practice_EN_block.saveAsExcel(filename + '.xlsx', sheetName='practice_EN_block',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # set up handler to look after randomisation of conditions etc
    practice_SWITCHING_block = data.TrialHandler2(
        name='practice_SWITCHING_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition/condition_practice_SWITCHING.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(practice_SWITCHING_block)  # add the loop to the experiment
    thisPractice_SWITCHING_block = practice_SWITCHING_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisPractice_SWITCHING_block.rgb)
    if thisPractice_SWITCHING_block != None:
        for paramName in thisPractice_SWITCHING_block:
            globals()[paramName] = thisPractice_SWITCHING_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisPractice_SWITCHING_block in practice_SWITCHING_block:
        practice_SWITCHING_block.status = STARTED
        if hasattr(thisPractice_SWITCHING_block, 'status'):
            thisPractice_SWITCHING_block.status = STARTED
        currentLoop = practice_SWITCHING_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisPractice_SWITCHING_block.rgb)
        if thisPractice_SWITCHING_block != None:
            for paramName in thisPractice_SWITCHING_block:
                globals()[paramName] = thisPractice_SWITCHING_block[paramName]
        
        # --- Prepare to start Routine "practice_dot_SWITCHING" ---
        # create an object to store info about Routine practice_dot_SWITCHING
        practice_dot_SWITCHING = data.Routine(
            name='practice_dot_SWITCHING',
            components=[practice_fixationdot_SWITCH],
        )
        practice_dot_SWITCHING.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for practice_dot_SWITCHING
        practice_dot_SWITCHING.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_dot_SWITCHING.tStart = globalClock.getTime(format='float')
        practice_dot_SWITCHING.status = STARTED
        thisExp.addData('practice_dot_SWITCHING.started', practice_dot_SWITCHING.tStart)
        practice_dot_SWITCHING.maxDuration = None
        # keep track of which components have finished
        practice_dot_SWITCHINGComponents = practice_dot_SWITCHING.components
        for thisComponent in practice_dot_SWITCHING.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_dot_SWITCHING" ---
        thisExp.currentRoutine = practice_dot_SWITCHING
        practice_dot_SWITCHING.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_SWITCHING_block, 'status') and thisPractice_SWITCHING_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_fixationdot_SWITCH* updates
            
            # if practice_fixationdot_SWITCH is starting this frame...
            if practice_fixationdot_SWITCH.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_fixationdot_SWITCH.frameNStart = frameN  # exact frame index
                practice_fixationdot_SWITCH.tStart = t  # local t and not account for scr refresh
                practice_fixationdot_SWITCH.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_fixationdot_SWITCH, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_fixationdot_SWITCH.started')
                # update status
                practice_fixationdot_SWITCH.status = STARTED
                practice_fixationdot_SWITCH.setAutoDraw(True)
            
            # if practice_fixationdot_SWITCH is active this frame...
            if practice_fixationdot_SWITCH.status == STARTED:
                # update params
                pass
            
            # if practice_fixationdot_SWITCH is stopping this frame...
            if practice_fixationdot_SWITCH.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_fixationdot_SWITCH.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_fixationdot_SWITCH.tStop = t  # not accounting for scr refresh
                    practice_fixationdot_SWITCH.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_fixationdot_SWITCH.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_fixationdot_SWITCH.stopped')
                    # update status
                    practice_fixationdot_SWITCH.status = FINISHED
                    practice_fixationdot_SWITCH.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_dot_SWITCHING,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_dot_SWITCHING.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_dot_SWITCHING.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_dot_SWITCHING.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_dot_SWITCHING" ---
        for thisComponent in practice_dot_SWITCHING.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_dot_SWITCHING
        practice_dot_SWITCHING.tStop = globalClock.getTime(format='float')
        practice_dot_SWITCHING.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_dot_SWITCHING.stopped', practice_dot_SWITCHING.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_dot_SWITCHING.maxDurationReached:
            routineTimer.addTime(-practice_dot_SWITCHING.maxDuration)
        elif practice_dot_SWITCHING.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "practice_picstim_SWITCHING" ---
        # create an object to store info about Routine practice_picstim_SWITCHING
        practice_picstim_SWITCHING = data.Routine(
            name='practice_picstim_SWITCHING',
            components=[practice_languagecue_SWITCHING, practice_picturestimuli_SWITCHING, practice_SWITCHING_response],
        )
        practice_picstim_SWITCHING.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        practice_languagecue_SWITCHING.setImage(path_lc_practice_SWITCHING)
        practice_picturestimuli_SWITCHING.setImage(path_ps_practice_SWITCHING)
        practice_SWITCHING_response.setPolicyWhenFull('warn')
        # store start times for practice_picstim_SWITCHING
        practice_picstim_SWITCHING.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_picstim_SWITCHING.tStart = globalClock.getTime(format='float')
        practice_picstim_SWITCHING.status = STARTED
        thisExp.addData('practice_picstim_SWITCHING.started', practice_picstim_SWITCHING.tStart)
        practice_picstim_SWITCHING.maxDuration = None
        # keep track of which components have finished
        practice_picstim_SWITCHINGComponents = practice_picstim_SWITCHING.components
        for thisComponent in practice_picstim_SWITCHING.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_picstim_SWITCHING" ---
        thisExp.currentRoutine = practice_picstim_SWITCHING
        practice_picstim_SWITCHING.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_SWITCHING_block, 'status') and thisPractice_SWITCHING_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_languagecue_SWITCHING* updates
            
            # if practice_languagecue_SWITCHING is starting this frame...
            if practice_languagecue_SWITCHING.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_languagecue_SWITCHING.frameNStart = frameN  # exact frame index
                practice_languagecue_SWITCHING.tStart = t  # local t and not account for scr refresh
                practice_languagecue_SWITCHING.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_languagecue_SWITCHING, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_languagecue_SWITCHING.started')
                # update status
                practice_languagecue_SWITCHING.status = STARTED
                practice_languagecue_SWITCHING.setAutoDraw(True)
            
            # if practice_languagecue_SWITCHING is active this frame...
            if practice_languagecue_SWITCHING.status == STARTED:
                # update params
                pass
            
            # if practice_languagecue_SWITCHING is stopping this frame...
            if practice_languagecue_SWITCHING.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_languagecue_SWITCHING.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_languagecue_SWITCHING.tStop = t  # not accounting for scr refresh
                    practice_languagecue_SWITCHING.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_languagecue_SWITCHING.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_languagecue_SWITCHING.stopped')
                    # update status
                    practice_languagecue_SWITCHING.status = FINISHED
                    practice_languagecue_SWITCHING.setAutoDraw(False)
            
            # *practice_picturestimuli_SWITCHING* updates
            
            # if practice_picturestimuli_SWITCHING is starting this frame...
            if practice_picturestimuli_SWITCHING.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_picturestimuli_SWITCHING.frameNStart = frameN  # exact frame index
                practice_picturestimuli_SWITCHING.tStart = t  # local t and not account for scr refresh
                practice_picturestimuli_SWITCHING.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_picturestimuli_SWITCHING, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_picturestimuli_SWITCHING.started')
                # update status
                practice_picturestimuli_SWITCHING.status = STARTED
                practice_picturestimuli_SWITCHING.setAutoDraw(True)
            
            # if practice_picturestimuli_SWITCHING is active this frame...
            if practice_picturestimuli_SWITCHING.status == STARTED:
                # update params
                pass
            
            # if practice_picturestimuli_SWITCHING is stopping this frame...
            if practice_picturestimuli_SWITCHING.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_picturestimuli_SWITCHING.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_picturestimuli_SWITCHING.tStop = t  # not accounting for scr refresh
                    practice_picturestimuli_SWITCHING.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_picturestimuli_SWITCHING.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_picturestimuli_SWITCHING.stopped')
                    # update status
                    practice_picturestimuli_SWITCHING.status = FINISHED
                    practice_picturestimuli_SWITCHING.setAutoDraw(False)
            
            # if practice_SWITCHING_response is starting this frame...
            if practice_SWITCHING_response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_SWITCHING_response.frameNStart = frameN  # exact frame index
                practice_SWITCHING_response.tStart = t  # local t and not account for scr refresh
                practice_SWITCHING_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_SWITCHING_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('practice_SWITCHING_response.started', t)
                # update status
                practice_SWITCHING_response.status = STARTED
                # start recording with practice_SWITCHING_response
                practice_SWITCHING_response.start()
            
            # if practice_SWITCHING_response is active this frame...
            if practice_SWITCHING_response.status == STARTED:
                # update params
                pass
                # update recorded clip for practice_SWITCHING_response
                practice_SWITCHING_response.poll()
            
            # if practice_SWITCHING_response is stopping this frame...
            if practice_SWITCHING_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_SWITCHING_response.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_SWITCHING_response.tStop = t  # not accounting for scr refresh
                    practice_SWITCHING_response.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_SWITCHING_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('practice_SWITCHING_response.stopped', t)
                    # update status
                    practice_SWITCHING_response.status = FINISHED
                    # stop recording with practice_SWITCHING_response
                    practice_SWITCHING_response.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_picstim_SWITCHING,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_picstim_SWITCHING.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_picstim_SWITCHING.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_picstim_SWITCHING.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_picstim_SWITCHING" ---
        for thisComponent in practice_picstim_SWITCHING.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_picstim_SWITCHING
        practice_picstim_SWITCHING.tStop = globalClock.getTime(format='float')
        practice_picstim_SWITCHING.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_picstim_SWITCHING.stopped', practice_picstim_SWITCHING.tStop)
        # tell mic to keep hold of current recording in practice_SWITCHING_response.clips and transcript (if applicable) in practice_SWITCHING_response.scripts
        # this will also update practice_SWITCHING_response.lastClip and practice_SWITCHING_response.lastScript
        practice_SWITCHING_response.stop()
        tag = data.utils.getDateStr()
        practice_SWITCHING_responseClip = practice_SWITCHING_response.bank(
            tag=tag, transcribe='None',
            config=None
        )
        practice_SWITCHING_block.addData(
            'practice_SWITCHING_response.clip', practice_SWITCHING_response.recordingFolder / practice_SWITCHING_response.getClipFilename(tag)
        )
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_picstim_SWITCHING.maxDurationReached:
            routineTimer.addTime(-practice_picstim_SWITCHING.maxDuration)
        elif practice_picstim_SWITCHING.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "practice_blank_SWITCHING" ---
        # create an object to store info about Routine practice_blank_SWITCHING
        practice_blank_SWITCHING = data.Routine(
            name='practice_blank_SWITCHING',
            components=[practice_blankscreen_SWITCHING],
        )
        practice_blank_SWITCHING.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for practice_blank_SWITCHING
        practice_blank_SWITCHING.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        practice_blank_SWITCHING.tStart = globalClock.getTime(format='float')
        practice_blank_SWITCHING.status = STARTED
        thisExp.addData('practice_blank_SWITCHING.started', practice_blank_SWITCHING.tStart)
        practice_blank_SWITCHING.maxDuration = None
        # keep track of which components have finished
        practice_blank_SWITCHINGComponents = practice_blank_SWITCHING.components
        for thisComponent in practice_blank_SWITCHING.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "practice_blank_SWITCHING" ---
        thisExp.currentRoutine = practice_blank_SWITCHING
        practice_blank_SWITCHING.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisPractice_SWITCHING_block, 'status') and thisPractice_SWITCHING_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *practice_blankscreen_SWITCHING* updates
            
            # if practice_blankscreen_SWITCHING is starting this frame...
            if practice_blankscreen_SWITCHING.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                practice_blankscreen_SWITCHING.frameNStart = frameN  # exact frame index
                practice_blankscreen_SWITCHING.tStart = t  # local t and not account for scr refresh
                practice_blankscreen_SWITCHING.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(practice_blankscreen_SWITCHING, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'practice_blankscreen_SWITCHING.started')
                # update status
                practice_blankscreen_SWITCHING.status = STARTED
                practice_blankscreen_SWITCHING.setAutoDraw(True)
            
            # if practice_blankscreen_SWITCHING is active this frame...
            if practice_blankscreen_SWITCHING.status == STARTED:
                # update params
                pass
            
            # if practice_blankscreen_SWITCHING is stopping this frame...
            if practice_blankscreen_SWITCHING.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > practice_blankscreen_SWITCHING.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    practice_blankscreen_SWITCHING.tStop = t  # not accounting for scr refresh
                    practice_blankscreen_SWITCHING.tStopRefresh = tThisFlipGlobal  # on global time
                    practice_blankscreen_SWITCHING.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'practice_blankscreen_SWITCHING.stopped')
                    # update status
                    practice_blankscreen_SWITCHING.status = FINISHED
                    practice_blankscreen_SWITCHING.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=practice_blank_SWITCHING,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                practice_blank_SWITCHING.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if practice_blank_SWITCHING.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in practice_blank_SWITCHING.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "practice_blank_SWITCHING" ---
        for thisComponent in practice_blank_SWITCHING.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for practice_blank_SWITCHING
        practice_blank_SWITCHING.tStop = globalClock.getTime(format='float')
        practice_blank_SWITCHING.tStopRefresh = tThisFlipGlobal
        thisExp.addData('practice_blank_SWITCHING.stopped', practice_blank_SWITCHING.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if practice_blank_SWITCHING.maxDurationReached:
            routineTimer.addTime(-practice_blank_SWITCHING.maxDuration)
        elif practice_blank_SWITCHING.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisPractice_SWITCHING_block as finished
        if hasattr(thisPractice_SWITCHING_block, 'status'):
            thisPractice_SWITCHING_block.status = FINISHED
        # if awaiting a pause, pause now
        if practice_SWITCHING_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            practice_SWITCHING_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'practice_SWITCHING_block'
    practice_SWITCHING_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if practice_SWITCHING_block.trialList in ([], [None], None):
        params = []
    else:
        params = practice_SWITCHING_block.trialList[0].keys()
    # save data for this loop
    practice_SWITCHING_block.saveAsExcel(filename + '.xlsx', sheetName='practice_SWITCHING_block',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "nonswitching_block" ---
    # create an object to store info about Routine nonswitching_block
    nonswitching_block = data.Routine(
        name='nonswitching_block',
        components=[text_nonswitching_block],
    )
    nonswitching_block.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code
    breakDuration = 31
    countdown = breakDuration
    # store start times for nonswitching_block
    nonswitching_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    nonswitching_block.tStart = globalClock.getTime(format='float')
    nonswitching_block.status = STARTED
    thisExp.addData('nonswitching_block.started', nonswitching_block.tStart)
    nonswitching_block.maxDuration = None
    # keep track of which components have finished
    nonswitching_blockComponents = nonswitching_block.components
    for thisComponent in nonswitching_block.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "nonswitching_block" ---
    thisExp.currentRoutine = nonswitching_block
    nonswitching_block.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_nonswitching_block* updates
        
        # if text_nonswitching_block is starting this frame...
        if text_nonswitching_block.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_nonswitching_block.frameNStart = frameN  # exact frame index
            text_nonswitching_block.tStart = t  # local t and not account for scr refresh
            text_nonswitching_block.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_nonswitching_block, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_nonswitching_block.started')
            # update status
            text_nonswitching_block.status = STARTED
            text_nonswitching_block.setAutoDraw(True)
        
        # if text_nonswitching_block is active this frame...
        if text_nonswitching_block.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from code
        remaining = int(breakDuration - t)
        countdown = max(remaining, 0)
        
        text_nonswitching_block.setText(
            "Selamat, Anda telah menyelesaikan blok latihan!\n\n"
            "Setelah jeda istirahat, Anda akan mengerjakan blok non-switching.\n"
            "Jangan lupa mengikuti kode bendera Indonesia dan Inggris.\n\n"
            f"Jeda istirahat dalam {countdown} detik"
        )
        
        if t >= breakDuration:
            continueRoutine = False
           
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=nonswitching_block,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            nonswitching_block.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if nonswitching_block.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in nonswitching_block.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "nonswitching_block" ---
    for thisComponent in nonswitching_block.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for nonswitching_block
    nonswitching_block.tStop = globalClock.getTime(format='float')
    nonswitching_block.tStopRefresh = tThisFlipGlobal
    thisExp.addData('nonswitching_block.stopped', nonswitching_block.tStop)
    thisExp.nextEntry()
    # the Routine "nonswitching_block" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    ID_block = data.TrialHandler2(
        name='ID_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition/condition_ID.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(ID_block)  # add the loop to the experiment
    thisID_block = ID_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisID_block.rgb)
    if thisID_block != None:
        for paramName in thisID_block:
            globals()[paramName] = thisID_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisID_block in ID_block:
        ID_block.status = STARTED
        if hasattr(thisID_block, 'status'):
            thisID_block.status = STARTED
        currentLoop = ID_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisID_block.rgb)
        if thisID_block != None:
            for paramName in thisID_block:
                globals()[paramName] = thisID_block[paramName]
        
        # --- Prepare to start Routine "dot_ID" ---
        # create an object to store info about Routine dot_ID
        dot_ID = data.Routine(
            name='dot_ID',
            components=[fixationdot_ID],
        )
        dot_ID.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for dot_ID
        dot_ID.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dot_ID.tStart = globalClock.getTime(format='float')
        dot_ID.status = STARTED
        thisExp.addData('dot_ID.started', dot_ID.tStart)
        dot_ID.maxDuration = None
        # keep track of which components have finished
        dot_IDComponents = dot_ID.components
        for thisComponent in dot_ID.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dot_ID" ---
        thisExp.currentRoutine = dot_ID
        dot_ID.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisID_block, 'status') and thisID_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixationdot_ID* updates
            
            # if fixationdot_ID is starting this frame...
            if fixationdot_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationdot_ID.frameNStart = frameN  # exact frame index
                fixationdot_ID.tStart = t  # local t and not account for scr refresh
                fixationdot_ID.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationdot_ID, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixationdot_ID.started')
                # update status
                fixationdot_ID.status = STARTED
                fixationdot_ID.setAutoDraw(True)
            
            # if fixationdot_ID is active this frame...
            if fixationdot_ID.status == STARTED:
                # update params
                pass
            
            # if fixationdot_ID is stopping this frame...
            if fixationdot_ID.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationdot_ID.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixationdot_ID.tStop = t  # not accounting for scr refresh
                    fixationdot_ID.tStopRefresh = tThisFlipGlobal  # on global time
                    fixationdot_ID.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationdot_ID.stopped')
                    # update status
                    fixationdot_ID.status = FINISHED
                    fixationdot_ID.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=dot_ID,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                dot_ID.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if dot_ID.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in dot_ID.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dot_ID" ---
        for thisComponent in dot_ID.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dot_ID
        dot_ID.tStop = globalClock.getTime(format='float')
        dot_ID.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dot_ID.stopped', dot_ID.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if dot_ID.maxDurationReached:
            routineTimer.addTime(-dot_ID.maxDuration)
        elif dot_ID.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "picstim_ID" ---
        # create an object to store info about Routine picstim_ID
        picstim_ID = data.Routine(
            name='picstim_ID',
            components=[languagecue_ID, picturestimuli_ID, ID_response],
        )
        picstim_ID.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        languagecue_ID.setImage(path_lc_ID)
        picturestimuli_ID.setImage(path_ps_ID)
        ID_response.setPolicyWhenFull('warn')
        # store start times for picstim_ID
        picstim_ID.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        picstim_ID.tStart = globalClock.getTime(format='float')
        picstim_ID.status = STARTED
        thisExp.addData('picstim_ID.started', picstim_ID.tStart)
        picstim_ID.maxDuration = None
        # keep track of which components have finished
        picstim_IDComponents = picstim_ID.components
        for thisComponent in picstim_ID.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "picstim_ID" ---
        thisExp.currentRoutine = picstim_ID
        picstim_ID.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisID_block, 'status') and thisID_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *languagecue_ID* updates
            
            # if languagecue_ID is starting this frame...
            if languagecue_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                languagecue_ID.frameNStart = frameN  # exact frame index
                languagecue_ID.tStart = t  # local t and not account for scr refresh
                languagecue_ID.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(languagecue_ID, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'languagecue_ID.started')
                # update status
                languagecue_ID.status = STARTED
                languagecue_ID.setAutoDraw(True)
            
            # if languagecue_ID is active this frame...
            if languagecue_ID.status == STARTED:
                # update params
                pass
            
            # if languagecue_ID is stopping this frame...
            if languagecue_ID.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > languagecue_ID.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    languagecue_ID.tStop = t  # not accounting for scr refresh
                    languagecue_ID.tStopRefresh = tThisFlipGlobal  # on global time
                    languagecue_ID.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'languagecue_ID.stopped')
                    # update status
                    languagecue_ID.status = FINISHED
                    languagecue_ID.setAutoDraw(False)
            
            # *picturestimuli_ID* updates
            
            # if picturestimuli_ID is starting this frame...
            if picturestimuli_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                picturestimuli_ID.frameNStart = frameN  # exact frame index
                picturestimuli_ID.tStart = t  # local t and not account for scr refresh
                picturestimuli_ID.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(picturestimuli_ID, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'picturestimuli_ID.started')
                # update status
                picturestimuli_ID.status = STARTED
                picturestimuli_ID.setAutoDraw(True)
            
            # if picturestimuli_ID is active this frame...
            if picturestimuli_ID.status == STARTED:
                # update params
                pass
            
            # if picturestimuli_ID is stopping this frame...
            if picturestimuli_ID.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > picturestimuli_ID.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    picturestimuli_ID.tStop = t  # not accounting for scr refresh
                    picturestimuli_ID.tStopRefresh = tThisFlipGlobal  # on global time
                    picturestimuli_ID.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'picturestimuli_ID.stopped')
                    # update status
                    picturestimuli_ID.status = FINISHED
                    picturestimuli_ID.setAutoDraw(False)
            
            # if ID_response is starting this frame...
            if ID_response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                ID_response.frameNStart = frameN  # exact frame index
                ID_response.tStart = t  # local t and not account for scr refresh
                ID_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(ID_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('ID_response.started', t)
                # update status
                ID_response.status = STARTED
                # start recording with ID_response
                ID_response.start()
            
            # if ID_response is active this frame...
            if ID_response.status == STARTED:
                # update params
                pass
                # update recorded clip for ID_response
                ID_response.poll()
            
            # if ID_response is stopping this frame...
            if ID_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > ID_response.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    ID_response.tStop = t  # not accounting for scr refresh
                    ID_response.tStopRefresh = tThisFlipGlobal  # on global time
                    ID_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('ID_response.stopped', t)
                    # update status
                    ID_response.status = FINISHED
                    # stop recording with ID_response
                    ID_response.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=picstim_ID,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                picstim_ID.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if picstim_ID.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in picstim_ID.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "picstim_ID" ---
        for thisComponent in picstim_ID.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for picstim_ID
        picstim_ID.tStop = globalClock.getTime(format='float')
        picstim_ID.tStopRefresh = tThisFlipGlobal
        thisExp.addData('picstim_ID.stopped', picstim_ID.tStop)
        # tell mic to keep hold of current recording in ID_response.clips and transcript (if applicable) in ID_response.scripts
        # this will also update ID_response.lastClip and ID_response.lastScript
        ID_response.stop()
        tag = data.utils.getDateStr()
        ID_responseClip = ID_response.bank(
            tag=tag, transcribe='None',
            config=None
        )
        ID_block.addData(
            'ID_response.clip', ID_response.recordingFolder / ID_response.getClipFilename(tag)
        )
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if picstim_ID.maxDurationReached:
            routineTimer.addTime(-picstim_ID.maxDuration)
        elif picstim_ID.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "blank_ID" ---
        # create an object to store info about Routine blank_ID
        blank_ID = data.Routine(
            name='blank_ID',
            components=[blankscreen_ID],
        )
        blank_ID.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank_ID
        blank_ID.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank_ID.tStart = globalClock.getTime(format='float')
        blank_ID.status = STARTED
        thisExp.addData('blank_ID.started', blank_ID.tStart)
        blank_ID.maxDuration = None
        # keep track of which components have finished
        blank_IDComponents = blank_ID.components
        for thisComponent in blank_ID.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_ID" ---
        thisExp.currentRoutine = blank_ID
        blank_ID.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisID_block, 'status') and thisID_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blankscreen_ID* updates
            
            # if blankscreen_ID is starting this frame...
            if blankscreen_ID.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blankscreen_ID.frameNStart = frameN  # exact frame index
                blankscreen_ID.tStart = t  # local t and not account for scr refresh
                blankscreen_ID.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blankscreen_ID, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blankscreen_ID.started')
                # update status
                blankscreen_ID.status = STARTED
                blankscreen_ID.setAutoDraw(True)
            
            # if blankscreen_ID is active this frame...
            if blankscreen_ID.status == STARTED:
                # update params
                pass
            
            # if blankscreen_ID is stopping this frame...
            if blankscreen_ID.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blankscreen_ID.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    blankscreen_ID.tStop = t  # not accounting for scr refresh
                    blankscreen_ID.tStopRefresh = tThisFlipGlobal  # on global time
                    blankscreen_ID.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blankscreen_ID.stopped')
                    # update status
                    blankscreen_ID.status = FINISHED
                    blankscreen_ID.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank_ID,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank_ID.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank_ID.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank_ID.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_ID" ---
        for thisComponent in blank_ID.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank_ID
        blank_ID.tStop = globalClock.getTime(format='float')
        blank_ID.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank_ID.stopped', blank_ID.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank_ID.maxDurationReached:
            routineTimer.addTime(-blank_ID.maxDuration)
        elif blank_ID.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisID_block as finished
        if hasattr(thisID_block, 'status'):
            thisID_block.status = FINISHED
        # if awaiting a pause, pause now
        if ID_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            ID_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'ID_block'
    ID_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if ID_block.trialList in ([], [None], None):
        params = []
    else:
        params = ID_block.trialList[0].keys()
    # save data for this loop
    ID_block.saveAsExcel(filename + '.xlsx', sheetName='ID_block',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # set up handler to look after randomisation of conditions etc
    EN_block = data.TrialHandler2(
        name='EN_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition/condition_EN.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(EN_block)  # add the loop to the experiment
    thisEN_block = EN_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisEN_block.rgb)
    if thisEN_block != None:
        for paramName in thisEN_block:
            globals()[paramName] = thisEN_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisEN_block in EN_block:
        EN_block.status = STARTED
        if hasattr(thisEN_block, 'status'):
            thisEN_block.status = STARTED
        currentLoop = EN_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisEN_block.rgb)
        if thisEN_block != None:
            for paramName in thisEN_block:
                globals()[paramName] = thisEN_block[paramName]
        
        # --- Prepare to start Routine "dot_EN" ---
        # create an object to store info about Routine dot_EN
        dot_EN = data.Routine(
            name='dot_EN',
            components=[fixationdot_EN],
        )
        dot_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for dot_EN
        dot_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dot_EN.tStart = globalClock.getTime(format='float')
        dot_EN.status = STARTED
        thisExp.addData('dot_EN.started', dot_EN.tStart)
        dot_EN.maxDuration = None
        # keep track of which components have finished
        dot_ENComponents = dot_EN.components
        for thisComponent in dot_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dot_EN" ---
        thisExp.currentRoutine = dot_EN
        dot_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisEN_block, 'status') and thisEN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixationdot_EN* updates
            
            # if fixationdot_EN is starting this frame...
            if fixationdot_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationdot_EN.frameNStart = frameN  # exact frame index
                fixationdot_EN.tStart = t  # local t and not account for scr refresh
                fixationdot_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationdot_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixationdot_EN.started')
                # update status
                fixationdot_EN.status = STARTED
                fixationdot_EN.setAutoDraw(True)
            
            # if fixationdot_EN is active this frame...
            if fixationdot_EN.status == STARTED:
                # update params
                pass
            
            # if fixationdot_EN is stopping this frame...
            if fixationdot_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationdot_EN.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixationdot_EN.tStop = t  # not accounting for scr refresh
                    fixationdot_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    fixationdot_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationdot_EN.stopped')
                    # update status
                    fixationdot_EN.status = FINISHED
                    fixationdot_EN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=dot_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                dot_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if dot_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in dot_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dot_EN" ---
        for thisComponent in dot_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dot_EN
        dot_EN.tStop = globalClock.getTime(format='float')
        dot_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dot_EN.stopped', dot_EN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if dot_EN.maxDurationReached:
            routineTimer.addTime(-dot_EN.maxDuration)
        elif dot_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "picstim_EN" ---
        # create an object to store info about Routine picstim_EN
        picstim_EN = data.Routine(
            name='picstim_EN',
            components=[languagecue_EN, picturestimuli_EN, EN_response],
        )
        picstim_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        languagecue_EN.setImage(path_lc_EN)
        picturestimuli_EN.setImage(path_ps_EN)
        EN_response.setPolicyWhenFull('warn')
        # store start times for picstim_EN
        picstim_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        picstim_EN.tStart = globalClock.getTime(format='float')
        picstim_EN.status = STARTED
        thisExp.addData('picstim_EN.started', picstim_EN.tStart)
        picstim_EN.maxDuration = None
        # keep track of which components have finished
        picstim_ENComponents = picstim_EN.components
        for thisComponent in picstim_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "picstim_EN" ---
        thisExp.currentRoutine = picstim_EN
        picstim_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisEN_block, 'status') and thisEN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *languagecue_EN* updates
            
            # if languagecue_EN is starting this frame...
            if languagecue_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                languagecue_EN.frameNStart = frameN  # exact frame index
                languagecue_EN.tStart = t  # local t and not account for scr refresh
                languagecue_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(languagecue_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'languagecue_EN.started')
                # update status
                languagecue_EN.status = STARTED
                languagecue_EN.setAutoDraw(True)
            
            # if languagecue_EN is active this frame...
            if languagecue_EN.status == STARTED:
                # update params
                pass
            
            # if languagecue_EN is stopping this frame...
            if languagecue_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > languagecue_EN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    languagecue_EN.tStop = t  # not accounting for scr refresh
                    languagecue_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    languagecue_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'languagecue_EN.stopped')
                    # update status
                    languagecue_EN.status = FINISHED
                    languagecue_EN.setAutoDraw(False)
            
            # *picturestimuli_EN* updates
            
            # if picturestimuli_EN is starting this frame...
            if picturestimuli_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                picturestimuli_EN.frameNStart = frameN  # exact frame index
                picturestimuli_EN.tStart = t  # local t and not account for scr refresh
                picturestimuli_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(picturestimuli_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'picturestimuli_EN.started')
                # update status
                picturestimuli_EN.status = STARTED
                picturestimuli_EN.setAutoDraw(True)
            
            # if picturestimuli_EN is active this frame...
            if picturestimuli_EN.status == STARTED:
                # update params
                pass
            
            # if picturestimuli_EN is stopping this frame...
            if picturestimuli_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > picturestimuli_EN.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    picturestimuli_EN.tStop = t  # not accounting for scr refresh
                    picturestimuli_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    picturestimuli_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'picturestimuli_EN.stopped')
                    # update status
                    picturestimuli_EN.status = FINISHED
                    picturestimuli_EN.setAutoDraw(False)
            
            # if EN_response is starting this frame...
            if EN_response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                EN_response.frameNStart = frameN  # exact frame index
                EN_response.tStart = t  # local t and not account for scr refresh
                EN_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(EN_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('EN_response.started', t)
                # update status
                EN_response.status = STARTED
                # start recording with EN_response
                EN_response.start()
            
            # if EN_response is active this frame...
            if EN_response.status == STARTED:
                # update params
                pass
                # update recorded clip for EN_response
                EN_response.poll()
            
            # if EN_response is stopping this frame...
            if EN_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > EN_response.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    EN_response.tStop = t  # not accounting for scr refresh
                    EN_response.tStopRefresh = tThisFlipGlobal  # on global time
                    EN_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('EN_response.stopped', t)
                    # update status
                    EN_response.status = FINISHED
                    # stop recording with EN_response
                    EN_response.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=picstim_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                picstim_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if picstim_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in picstim_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "picstim_EN" ---
        for thisComponent in picstim_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for picstim_EN
        picstim_EN.tStop = globalClock.getTime(format='float')
        picstim_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('picstim_EN.stopped', picstim_EN.tStop)
        # tell mic to keep hold of current recording in EN_response.clips and transcript (if applicable) in EN_response.scripts
        # this will also update EN_response.lastClip and EN_response.lastScript
        EN_response.stop()
        tag = data.utils.getDateStr()
        EN_responseClip = EN_response.bank(
            tag=tag, transcribe='None',
            config=None
        )
        EN_block.addData(
            'EN_response.clip', EN_response.recordingFolder / EN_response.getClipFilename(tag)
        )
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if picstim_EN.maxDurationReached:
            routineTimer.addTime(-picstim_EN.maxDuration)
        elif picstim_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "blank_EN" ---
        # create an object to store info about Routine blank_EN
        blank_EN = data.Routine(
            name='blank_EN',
            components=[blankscreen_EN],
        )
        blank_EN.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank_EN
        blank_EN.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank_EN.tStart = globalClock.getTime(format='float')
        blank_EN.status = STARTED
        thisExp.addData('blank_EN.started', blank_EN.tStart)
        blank_EN.maxDuration = None
        # keep track of which components have finished
        blank_ENComponents = blank_EN.components
        for thisComponent in blank_EN.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_EN" ---
        thisExp.currentRoutine = blank_EN
        blank_EN.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisEN_block, 'status') and thisEN_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blankscreen_EN* updates
            
            # if blankscreen_EN is starting this frame...
            if blankscreen_EN.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blankscreen_EN.frameNStart = frameN  # exact frame index
                blankscreen_EN.tStart = t  # local t and not account for scr refresh
                blankscreen_EN.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blankscreen_EN, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blankscreen_EN.started')
                # update status
                blankscreen_EN.status = STARTED
                blankscreen_EN.setAutoDraw(True)
            
            # if blankscreen_EN is active this frame...
            if blankscreen_EN.status == STARTED:
                # update params
                pass
            
            # if blankscreen_EN is stopping this frame...
            if blankscreen_EN.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blankscreen_EN.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    blankscreen_EN.tStop = t  # not accounting for scr refresh
                    blankscreen_EN.tStopRefresh = tThisFlipGlobal  # on global time
                    blankscreen_EN.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blankscreen_EN.stopped')
                    # update status
                    blankscreen_EN.status = FINISHED
                    blankscreen_EN.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank_EN,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank_EN.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank_EN.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank_EN.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_EN" ---
        for thisComponent in blank_EN.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank_EN
        blank_EN.tStop = globalClock.getTime(format='float')
        blank_EN.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank_EN.stopped', blank_EN.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank_EN.maxDurationReached:
            routineTimer.addTime(-blank_EN.maxDuration)
        elif blank_EN.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisEN_block as finished
        if hasattr(thisEN_block, 'status'):
            thisEN_block.status = FINISHED
        # if awaiting a pause, pause now
        if EN_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            EN_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'EN_block'
    EN_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if EN_block.trialList in ([], [None], None):
        params = []
    else:
        params = EN_block.trialList[0].keys()
    # save data for this loop
    EN_block.saveAsExcel(filename + '.xlsx', sheetName='EN_block',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "switching_block" ---
    # create an object to store info about Routine switching_block
    switching_block = data.Routine(
        name='switching_block',
        components=[text_switching_block],
    )
    switching_block.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # Run 'Begin Routine' code from code_2
    breakDuration = 31
    countdown = breakDuration
    # store start times for switching_block
    switching_block.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    switching_block.tStart = globalClock.getTime(format='float')
    switching_block.status = STARTED
    thisExp.addData('switching_block.started', switching_block.tStart)
    switching_block.maxDuration = None
    # keep track of which components have finished
    switching_blockComponents = switching_block.components
    for thisComponent in switching_block.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "switching_block" ---
    thisExp.currentRoutine = switching_block
    switching_block.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text_switching_block* updates
        
        # if text_switching_block is starting this frame...
        if text_switching_block.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text_switching_block.frameNStart = frameN  # exact frame index
            text_switching_block.tStart = t  # local t and not account for scr refresh
            text_switching_block.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text_switching_block, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text_switching_block.started')
            # update status
            text_switching_block.status = STARTED
            text_switching_block.setAutoDraw(True)
        
        # if text_switching_block is active this frame...
        if text_switching_block.status == STARTED:
            # update params
            pass
        # Run 'Each Frame' code from code_2
        remaining = int(breakDuration - t)
        countdown = max(remaining, 0)
        
        text_switching_block.setText(
            "Selamat, Anda telah menyelesaikan blok non-switching!\n\n"
            "Setelah jeda istirahat, Anda akan mengerjakan blok switching.\n"
            "Ikuti kode bendera Indonesia dan Inggris seperti sebelumnya.\n\n"
            f"Jeda istirahat dalam {countdown} detik"
        )
        
        if t >= breakDuration:
            continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=switching_block,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            switching_block.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if switching_block.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in switching_block.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "switching_block" ---
    for thisComponent in switching_block.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for switching_block
    switching_block.tStop = globalClock.getTime(format='float')
    switching_block.tStopRefresh = tThisFlipGlobal
    thisExp.addData('switching_block.stopped', switching_block.tStop)
    thisExp.nextEntry()
    # the Routine "switching_block" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    
    # set up handler to look after randomisation of conditions etc
    SWITCHING_block = data.TrialHandler2(
        name='SWITCHING_block',
        nReps=1.0, 
        method='random', 
        extraInfo=expInfo, 
        originPath=-1, 
        trialList=data.importConditions('condition/condition_SWITCHING.xlsx'), 
        seed=None, 
        isTrials=True, 
    )
    thisExp.addLoop(SWITCHING_block)  # add the loop to the experiment
    thisSWITCHING_block = SWITCHING_block.trialList[0]  # so we can initialise stimuli with some values
    # abbreviate parameter names if possible (e.g. rgb = thisSWITCHING_block.rgb)
    if thisSWITCHING_block != None:
        for paramName in thisSWITCHING_block:
            globals()[paramName] = thisSWITCHING_block[paramName]
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    
    for thisSWITCHING_block in SWITCHING_block:
        SWITCHING_block.status = STARTED
        if hasattr(thisSWITCHING_block, 'status'):
            thisSWITCHING_block.status = STARTED
        currentLoop = SWITCHING_block
        thisExp.timestampOnFlip(win, 'thisRow.t', format=globalClock.format)
        if thisSession is not None:
            # if running in a Session with a Liaison client, send data up to now
            thisSession.sendExperimentData()
        # abbreviate parameter names if possible (e.g. rgb = thisSWITCHING_block.rgb)
        if thisSWITCHING_block != None:
            for paramName in thisSWITCHING_block:
                globals()[paramName] = thisSWITCHING_block[paramName]
        
        # --- Prepare to start Routine "dot_SWITCHING" ---
        # create an object to store info about Routine dot_SWITCHING
        dot_SWITCHING = data.Routine(
            name='dot_SWITCHING',
            components=[fixationdot_MIX],
        )
        dot_SWITCHING.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for dot_SWITCHING
        dot_SWITCHING.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        dot_SWITCHING.tStart = globalClock.getTime(format='float')
        dot_SWITCHING.status = STARTED
        thisExp.addData('dot_SWITCHING.started', dot_SWITCHING.tStart)
        dot_SWITCHING.maxDuration = None
        # keep track of which components have finished
        dot_SWITCHINGComponents = dot_SWITCHING.components
        for thisComponent in dot_SWITCHING.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "dot_SWITCHING" ---
        thisExp.currentRoutine = dot_SWITCHING
        dot_SWITCHING.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 0.5:
            # if trial has changed, end Routine now
            if hasattr(thisSWITCHING_block, 'status') and thisSWITCHING_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *fixationdot_MIX* updates
            
            # if fixationdot_MIX is starting this frame...
            if fixationdot_MIX.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                fixationdot_MIX.frameNStart = frameN  # exact frame index
                fixationdot_MIX.tStart = t  # local t and not account for scr refresh
                fixationdot_MIX.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(fixationdot_MIX, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'fixationdot_MIX.started')
                # update status
                fixationdot_MIX.status = STARTED
                fixationdot_MIX.setAutoDraw(True)
            
            # if fixationdot_MIX is active this frame...
            if fixationdot_MIX.status == STARTED:
                # update params
                pass
            
            # if fixationdot_MIX is stopping this frame...
            if fixationdot_MIX.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > fixationdot_MIX.tStartRefresh + 0.5-frameTolerance:
                    # keep track of stop time/frame for later
                    fixationdot_MIX.tStop = t  # not accounting for scr refresh
                    fixationdot_MIX.tStopRefresh = tThisFlipGlobal  # on global time
                    fixationdot_MIX.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'fixationdot_MIX.stopped')
                    # update status
                    fixationdot_MIX.status = FINISHED
                    fixationdot_MIX.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=dot_SWITCHING,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                dot_SWITCHING.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if dot_SWITCHING.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in dot_SWITCHING.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "dot_SWITCHING" ---
        for thisComponent in dot_SWITCHING.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for dot_SWITCHING
        dot_SWITCHING.tStop = globalClock.getTime(format='float')
        dot_SWITCHING.tStopRefresh = tThisFlipGlobal
        thisExp.addData('dot_SWITCHING.stopped', dot_SWITCHING.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if dot_SWITCHING.maxDurationReached:
            routineTimer.addTime(-dot_SWITCHING.maxDuration)
        elif dot_SWITCHING.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-0.500000)
        
        # --- Prepare to start Routine "picstim_SWITCHING" ---
        # create an object to store info about Routine picstim_SWITCHING
        picstim_SWITCHING = data.Routine(
            name='picstim_SWITCHING',
            components=[languagecue_SWITCHING, picturestimuli_SWITCHING, SWITCHING_response],
        )
        picstim_SWITCHING.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        languagecue_SWITCHING.setImage(path_lc_SWITCHING)
        picturestimuli_SWITCHING.setImage(path_ps_SWITCHING)
        SWITCHING_response.setPolicyWhenFull('warn')
        # store start times for picstim_SWITCHING
        picstim_SWITCHING.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        picstim_SWITCHING.tStart = globalClock.getTime(format='float')
        picstim_SWITCHING.status = STARTED
        thisExp.addData('picstim_SWITCHING.started', picstim_SWITCHING.tStart)
        picstim_SWITCHING.maxDuration = None
        # keep track of which components have finished
        picstim_SWITCHINGComponents = picstim_SWITCHING.components
        for thisComponent in picstim_SWITCHING.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "picstim_SWITCHING" ---
        thisExp.currentRoutine = picstim_SWITCHING
        picstim_SWITCHING.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 2.0:
            # if trial has changed, end Routine now
            if hasattr(thisSWITCHING_block, 'status') and thisSWITCHING_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *languagecue_SWITCHING* updates
            
            # if languagecue_SWITCHING is starting this frame...
            if languagecue_SWITCHING.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                languagecue_SWITCHING.frameNStart = frameN  # exact frame index
                languagecue_SWITCHING.tStart = t  # local t and not account for scr refresh
                languagecue_SWITCHING.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(languagecue_SWITCHING, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'languagecue_SWITCHING.started')
                # update status
                languagecue_SWITCHING.status = STARTED
                languagecue_SWITCHING.setAutoDraw(True)
            
            # if languagecue_SWITCHING is active this frame...
            if languagecue_SWITCHING.status == STARTED:
                # update params
                pass
            
            # if languagecue_SWITCHING is stopping this frame...
            if languagecue_SWITCHING.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > languagecue_SWITCHING.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    languagecue_SWITCHING.tStop = t  # not accounting for scr refresh
                    languagecue_SWITCHING.tStopRefresh = tThisFlipGlobal  # on global time
                    languagecue_SWITCHING.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'languagecue_SWITCHING.stopped')
                    # update status
                    languagecue_SWITCHING.status = FINISHED
                    languagecue_SWITCHING.setAutoDraw(False)
            
            # *picturestimuli_SWITCHING* updates
            
            # if picturestimuli_SWITCHING is starting this frame...
            if picturestimuli_SWITCHING.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                picturestimuli_SWITCHING.frameNStart = frameN  # exact frame index
                picturestimuli_SWITCHING.tStart = t  # local t and not account for scr refresh
                picturestimuli_SWITCHING.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(picturestimuli_SWITCHING, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'picturestimuli_SWITCHING.started')
                # update status
                picturestimuli_SWITCHING.status = STARTED
                picturestimuli_SWITCHING.setAutoDraw(True)
            
            # if picturestimuli_SWITCHING is active this frame...
            if picturestimuli_SWITCHING.status == STARTED:
                # update params
                pass
            
            # if picturestimuli_SWITCHING is stopping this frame...
            if picturestimuli_SWITCHING.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > picturestimuli_SWITCHING.tStartRefresh + 2-frameTolerance:
                    # keep track of stop time/frame for later
                    picturestimuli_SWITCHING.tStop = t  # not accounting for scr refresh
                    picturestimuli_SWITCHING.tStopRefresh = tThisFlipGlobal  # on global time
                    picturestimuli_SWITCHING.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'picturestimuli_SWITCHING.stopped')
                    # update status
                    picturestimuli_SWITCHING.status = FINISHED
                    picturestimuli_SWITCHING.setAutoDraw(False)
            
            # if SWITCHING_response is starting this frame...
            if SWITCHING_response.status == NOT_STARTED and t >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                SWITCHING_response.frameNStart = frameN  # exact frame index
                SWITCHING_response.tStart = t  # local t and not account for scr refresh
                SWITCHING_response.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(SWITCHING_response, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.addData('SWITCHING_response.started', t)
                # update status
                SWITCHING_response.status = STARTED
                # start recording with SWITCHING_response
                SWITCHING_response.start()
            
            # if SWITCHING_response is active this frame...
            if SWITCHING_response.status == STARTED:
                # update params
                pass
                # update recorded clip for SWITCHING_response
                SWITCHING_response.poll()
            
            # if SWITCHING_response is stopping this frame...
            if SWITCHING_response.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > SWITCHING_response.tStartRefresh + 2.0-frameTolerance:
                    # keep track of stop time/frame for later
                    SWITCHING_response.tStop = t  # not accounting for scr refresh
                    SWITCHING_response.tStopRefresh = tThisFlipGlobal  # on global time
                    SWITCHING_response.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.addData('SWITCHING_response.stopped', t)
                    # update status
                    SWITCHING_response.status = FINISHED
                    # stop recording with SWITCHING_response
                    SWITCHING_response.stop()
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=picstim_SWITCHING,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                picstim_SWITCHING.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if picstim_SWITCHING.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in picstim_SWITCHING.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "picstim_SWITCHING" ---
        for thisComponent in picstim_SWITCHING.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for picstim_SWITCHING
        picstim_SWITCHING.tStop = globalClock.getTime(format='float')
        picstim_SWITCHING.tStopRefresh = tThisFlipGlobal
        thisExp.addData('picstim_SWITCHING.stopped', picstim_SWITCHING.tStop)
        # tell mic to keep hold of current recording in SWITCHING_response.clips and transcript (if applicable) in SWITCHING_response.scripts
        # this will also update SWITCHING_response.lastClip and SWITCHING_response.lastScript
        SWITCHING_response.stop()
        tag = data.utils.getDateStr()
        SWITCHING_responseClip = SWITCHING_response.bank(
            tag=tag, transcribe='None',
            config=None
        )
        SWITCHING_block.addData(
            'SWITCHING_response.clip', SWITCHING_response.recordingFolder / SWITCHING_response.getClipFilename(tag)
        )
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if picstim_SWITCHING.maxDurationReached:
            routineTimer.addTime(-picstim_SWITCHING.maxDuration)
        elif picstim_SWITCHING.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-2.000000)
        
        # --- Prepare to start Routine "blank_SWITCHING" ---
        # create an object to store info about Routine blank_SWITCHING
        blank_SWITCHING = data.Routine(
            name='blank_SWITCHING',
            components=[blankscreen_SWITCH],
        )
        blank_SWITCHING.status = NOT_STARTED
        continueRoutine = True
        # update component parameters for each repeat
        # store start times for blank_SWITCHING
        blank_SWITCHING.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
        blank_SWITCHING.tStart = globalClock.getTime(format='float')
        blank_SWITCHING.status = STARTED
        thisExp.addData('blank_SWITCHING.started', blank_SWITCHING.tStart)
        blank_SWITCHING.maxDuration = None
        # keep track of which components have finished
        blank_SWITCHINGComponents = blank_SWITCHING.components
        for thisComponent in blank_SWITCHING.components:
            thisComponent.tStart = None
            thisComponent.tStop = None
            thisComponent.tStartRefresh = None
            thisComponent.tStopRefresh = None
            if hasattr(thisComponent, 'status'):
                thisComponent.status = NOT_STARTED
        # reset timers
        t = 0
        _timeToFirstFrame = win.getFutureFlipTime(clock="now")
        frameN = -1
        
        # --- Run Routine "blank_SWITCHING" ---
        thisExp.currentRoutine = blank_SWITCHING
        blank_SWITCHING.forceEnded = routineForceEnded = not continueRoutine
        while continueRoutine and routineTimer.getTime() < 1.0:
            # if trial has changed, end Routine now
            if hasattr(thisSWITCHING_block, 'status') and thisSWITCHING_block.status == STOPPING:
                continueRoutine = False
            # get current time
            t = routineTimer.getTime()
            tThisFlip = win.getFutureFlipTime(clock=routineTimer)
            tThisFlipGlobal = win.getFutureFlipTime(clock=None)
            frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
            # update/draw components on each frame
            
            # *blankscreen_SWITCH* updates
            
            # if blankscreen_SWITCH is starting this frame...
            if blankscreen_SWITCH.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
                # keep track of start time/frame for later
                blankscreen_SWITCH.frameNStart = frameN  # exact frame index
                blankscreen_SWITCH.tStart = t  # local t and not account for scr refresh
                blankscreen_SWITCH.tStartRefresh = tThisFlipGlobal  # on global time
                win.timeOnFlip(blankscreen_SWITCH, 'tStartRefresh')  # time at next scr refresh
                # add timestamp to datafile
                thisExp.timestampOnFlip(win, 'blankscreen_SWITCH.started')
                # update status
                blankscreen_SWITCH.status = STARTED
                blankscreen_SWITCH.setAutoDraw(True)
            
            # if blankscreen_SWITCH is active this frame...
            if blankscreen_SWITCH.status == STARTED:
                # update params
                pass
            
            # if blankscreen_SWITCH is stopping this frame...
            if blankscreen_SWITCH.status == STARTED:
                # is it time to stop? (based on global clock, using actual start)
                if tThisFlipGlobal > blankscreen_SWITCH.tStartRefresh + 1.0-frameTolerance:
                    # keep track of stop time/frame for later
                    blankscreen_SWITCH.tStop = t  # not accounting for scr refresh
                    blankscreen_SWITCH.tStopRefresh = tThisFlipGlobal  # on global time
                    blankscreen_SWITCH.frameNStop = frameN  # exact frame index
                    # add timestamp to datafile
                    thisExp.timestampOnFlip(win, 'blankscreen_SWITCH.stopped')
                    # update status
                    blankscreen_SWITCH.status = FINISHED
                    blankscreen_SWITCH.setAutoDraw(False)
            
            # check for quit (typically the Esc key)
            if defaultKeyboard.getKeys(keyList=["escape"]):
                thisExp.status = FINISHED
            if thisExp.status == FINISHED or endExpNow:
                endExperiment(thisExp, win=win)
                return
            # pause experiment here if requested
            if thisExp.status == PAUSED:
                pauseExperiment(
                    thisExp=thisExp, 
                    win=win, 
                    timers=[routineTimer, globalClock], 
                    currentRoutine=blank_SWITCHING,
                )
                # skip the frame we paused on
                continue
            
            # has a Component requested the Routine to end?
            if not continueRoutine:
                blank_SWITCHING.forceEnded = routineForceEnded = True
            # has the Routine been forcibly ended?
            if blank_SWITCHING.forceEnded or routineForceEnded:
                break
            # has every Component finished?
            continueRoutine = False
            for thisComponent in blank_SWITCHING.components:
                if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                    continueRoutine = True
                    break  # at least one component has not yet finished
            
            # refresh the screen
            if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
                win.flip()
        
        # --- Ending Routine "blank_SWITCHING" ---
        for thisComponent in blank_SWITCHING.components:
            if hasattr(thisComponent, "setAutoDraw"):
                thisComponent.setAutoDraw(False)
        # store stop times for blank_SWITCHING
        blank_SWITCHING.tStop = globalClock.getTime(format='float')
        blank_SWITCHING.tStopRefresh = tThisFlipGlobal
        thisExp.addData('blank_SWITCHING.stopped', blank_SWITCHING.tStop)
        # using non-slip timing so subtract the expected duration of this Routine (unless ended on request)
        if blank_SWITCHING.maxDurationReached:
            routineTimer.addTime(-blank_SWITCHING.maxDuration)
        elif blank_SWITCHING.forceEnded:
            routineTimer.reset()
        else:
            routineTimer.addTime(-1.000000)
        # mark thisSWITCHING_block as finished
        if hasattr(thisSWITCHING_block, 'status'):
            thisSWITCHING_block.status = FINISHED
        # if awaiting a pause, pause now
        if SWITCHING_block.status == PAUSED:
            thisExp.status = PAUSED
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[globalClock], 
            )
            # once done pausing, restore running status
            SWITCHING_block.status = STARTED
        thisExp.nextEntry()
        
    # completed 1.0 repeats of 'SWITCHING_block'
    SWITCHING_block.status = FINISHED
    
    if thisSession is not None:
        # if running in a Session with a Liaison client, send data up to now
        thisSession.sendExperimentData()
    # get names of stimulus parameters
    if SWITCHING_block.trialList in ([], [None], None):
        params = []
    else:
        params = SWITCHING_block.trialList[0].keys()
    # save data for this loop
    SWITCHING_block.saveAsExcel(filename + '.xlsx', sheetName='SWITCHING_block',
        stimOut=params,
        dataOut=['n','all_mean','all_std', 'all_raw'])
    
    # --- Prepare to start Routine "experiment_end" ---
    # create an object to store info about Routine experiment_end
    experiment_end = data.Routine(
        name='experiment_end',
        components=[text, key_resp_3],
    )
    experiment_end.status = NOT_STARTED
    continueRoutine = True
    # update component parameters for each repeat
    # create starting attributes for key_resp_3
    key_resp_3.keys = []
    key_resp_3.rt = []
    _key_resp_3_allKeys = []
    # store start times for experiment_end
    experiment_end.tStartRefresh = win.getFutureFlipTime(clock=globalClock)
    experiment_end.tStart = globalClock.getTime(format='float')
    experiment_end.status = STARTED
    thisExp.addData('experiment_end.started', experiment_end.tStart)
    experiment_end.maxDuration = None
    # keep track of which components have finished
    experiment_endComponents = experiment_end.components
    for thisComponent in experiment_end.components:
        thisComponent.tStart = None
        thisComponent.tStop = None
        thisComponent.tStartRefresh = None
        thisComponent.tStopRefresh = None
        if hasattr(thisComponent, 'status'):
            thisComponent.status = NOT_STARTED
    # reset timers
    t = 0
    _timeToFirstFrame = win.getFutureFlipTime(clock="now")
    frameN = -1
    
    # --- Run Routine "experiment_end" ---
    thisExp.currentRoutine = experiment_end
    experiment_end.forceEnded = routineForceEnded = not continueRoutine
    while continueRoutine:
        # get current time
        t = routineTimer.getTime()
        tThisFlip = win.getFutureFlipTime(clock=routineTimer)
        tThisFlipGlobal = win.getFutureFlipTime(clock=None)
        frameN = frameN + 1  # number of completed frames (so 0 is the first frame)
        # update/draw components on each frame
        
        # *text* updates
        
        # if text is starting this frame...
        if text.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            text.frameNStart = frameN  # exact frame index
            text.tStart = t  # local t and not account for scr refresh
            text.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(text, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'text.started')
            # update status
            text.status = STARTED
            text.setAutoDraw(True)
        
        # if text is active this frame...
        if text.status == STARTED:
            # update params
            pass
        
        # *key_resp_3* updates
        waitOnFlip = False
        
        # if key_resp_3 is starting this frame...
        if key_resp_3.status == NOT_STARTED and tThisFlip >= 0.0-frameTolerance:
            # keep track of start time/frame for later
            key_resp_3.frameNStart = frameN  # exact frame index
            key_resp_3.tStart = t  # local t and not account for scr refresh
            key_resp_3.tStartRefresh = tThisFlipGlobal  # on global time
            win.timeOnFlip(key_resp_3, 'tStartRefresh')  # time at next scr refresh
            # add timestamp to datafile
            thisExp.timestampOnFlip(win, 'key_resp_3.started')
            # update status
            key_resp_3.status = STARTED
            # keyboard checking is just starting
            waitOnFlip = True
            win.callOnFlip(key_resp_3.clock.reset)  # t=0 on next screen flip
            win.callOnFlip(key_resp_3.clearEvents, eventType='keyboard')  # clear events on next screen flip
        if key_resp_3.status == STARTED and not waitOnFlip:
            theseKeys = key_resp_3.getKeys(keyList=['space'], ignoreKeys=["escape"], waitRelease=False)
            _key_resp_3_allKeys.extend(theseKeys)
            if len(_key_resp_3_allKeys):
                key_resp_3.keys = _key_resp_3_allKeys[-1].name  # just the last key pressed
                key_resp_3.rt = _key_resp_3_allKeys[-1].rt
                key_resp_3.duration = _key_resp_3_allKeys[-1].duration
                # a response ends the routine
                continueRoutine = False
        
        # check for quit (typically the Esc key)
        if defaultKeyboard.getKeys(keyList=["escape"]):
            thisExp.status = FINISHED
        if thisExp.status == FINISHED or endExpNow:
            endExperiment(thisExp, win=win)
            return
        # pause experiment here if requested
        if thisExp.status == PAUSED:
            pauseExperiment(
                thisExp=thisExp, 
                win=win, 
                timers=[routineTimer, globalClock], 
                currentRoutine=experiment_end,
            )
            # skip the frame we paused on
            continue
        
        # has a Component requested the Routine to end?
        if not continueRoutine:
            experiment_end.forceEnded = routineForceEnded = True
        # has the Routine been forcibly ended?
        if experiment_end.forceEnded or routineForceEnded:
            break
        # has every Component finished?
        continueRoutine = False
        for thisComponent in experiment_end.components:
            if hasattr(thisComponent, "status") and thisComponent.status != FINISHED:
                continueRoutine = True
                break  # at least one component has not yet finished
        
        # refresh the screen
        if continueRoutine:  # don't flip if this routine is over or we'll get a blank screen
            win.flip()
    
    # --- Ending Routine "experiment_end" ---
    for thisComponent in experiment_end.components:
        if hasattr(thisComponent, "setAutoDraw"):
            thisComponent.setAutoDraw(False)
    # store stop times for experiment_end
    experiment_end.tStop = globalClock.getTime(format='float')
    experiment_end.tStopRefresh = tThisFlipGlobal
    thisExp.addData('experiment_end.stopped', experiment_end.tStop)
    # check responses
    if key_resp_3.keys in ['', [], None]:  # No response was made
        key_resp_3.keys = None
    thisExp.addData('key_resp_3.keys',key_resp_3.keys)
    if key_resp_3.keys != None:  # we had a response
        thisExp.addData('key_resp_3.rt', key_resp_3.rt)
        thisExp.addData('key_resp_3.duration', key_resp_3.duration)
    thisExp.nextEntry()
    # the Routine "experiment_end" was not non-slip safe, so reset the non-slip timer
    routineTimer.reset()
    # save practice_ID_response recordings
    practice_ID_response.saveClips()
    # save practice_EN_response recordings
    practice_EN_response.saveClips()
    # save practice_SWITCHING_response recordings
    practice_SWITCHING_response.saveClips()
    # save ID_response recordings
    ID_response.saveClips()
    # save EN_response recordings
    EN_response.saveClips()
    # save SWITCHING_response recordings
    SWITCHING_response.saveClips()
    
    # mark experiment as finished
    endExperiment(thisExp, win=win)


def saveData(thisExp):
    """
    Save data from this experiment
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    """
    filename = thisExp.dataFileName
    # these shouldn't be strictly necessary (should auto-save)
    thisExp.saveAsWideText(filename + '.csv', delim='auto')
    thisExp.saveAsPickle(filename)


def endExperiment(thisExp, win=None):
    """
    End this experiment, performing final shut down operations.
    
    This function does NOT close the window or end the Python process - use `quit` for this.
    
    Parameters
    ==========
    thisExp : psychopy.data.ExperimentHandler
        Handler object for this experiment, contains the data to save and information about 
        where to save it to.
    win : psychopy.visual.Window
        Window for this experiment.
    """
    if win is not None:
        # remove autodraw from all current components
        win.clearAutoDraw()
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed
        win.flip()
    # return console logger level to WARNING
    logging.console.setLevel(logging.WARNING)
    # mark experiment handler as finished
    thisExp.status = FINISHED
    # run any 'at exit' functions
    for fcn in runAtExit:
        fcn()
    logging.flush()


def quit(thisExp, win=None, thisSession=None):
    """
    Fully quit, closing the window and ending the Python process.
    
    Parameters
    ==========
    win : psychopy.visual.Window
        Window to close.
    thisSession : psychopy.session.Session or None
        Handle of the Session object this experiment is being run from, if any.
    """
    thisExp.abort()  # or data files will save again on exit
    # make sure everything is closed down
    if win is not None:
        # Flip one final time so any remaining win.callOnFlip() 
        # and win.timeOnFlip() tasks get executed before quitting
        win.flip()
        win.close()
    logging.flush()
    if thisSession is not None:
        thisSession.stop()
    # terminate Python process
    core.quit()


# if running this experiment as a script...
if __name__ == '__main__':
    # call all functions in order
    expInfo = showExpInfoDlg(expInfo=expInfo)
    thisExp = setupData(expInfo=expInfo)
    logFile = setupLogging(filename=thisExp.dataFileName)
    win = setupWindow(expInfo=expInfo)
    setupDevices(expInfo=expInfo, thisExp=thisExp, win=win)
    run(
        expInfo=expInfo, 
        thisExp=thisExp, 
        win=win,
        globalClock='float'
    )
    saveData(thisExp=thisExp)
    quit(thisExp=thisExp, win=win)
