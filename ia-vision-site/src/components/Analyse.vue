<template>
  <div
    v-if="analysisData.gender"
    class="w-[850px] h-[250px] bg-white/5 rounded-3xl p-6 flex justify-between items-center backdrop-blur-xs"
  >
    <!-- Section Genre -->
    <div class="flex flex-col items-center text-white">
      <h3 class="text-xl font-semibold mb-4 text-gray-300">Genre</h3>

      <div
        class="rounded-2xl p-6 w-56 h-40 flex flex-col items-center justify-center border border-white/30"
      >
        <!-- Icônes homme/femme -->
        <div class="flex gap-4 mb-3">
          <img
            v-if="analysisData.gender.genderName === 'Man'"
            src="/man.svg"
            alt="Homme"
            class="w-8 h-8 opacity-80"
          />
          <img v-else src="/woman.svg" alt="Femme" class="w-8 h-8 opacity-40" />
        </div>

        <!-- Texte principal -->
        <div class="text-center">
          <p class="text-xl font-bold mb-2">
            {{ analysisData.gender.genderName === 'Man' ? 'Homme' : 'Femme' }}
          </p>
          <p class="text-sm text-gray-400">Sur a :{{ analysisData.gender.percentage }}</p>
        </div>
      </div>
    </div>

    <!-- Section Age -->
    <div class="flex flex-col items-center text-white">
      <h3 class="text-xl font-semibold mb-4 text-gray-300">Age</h3>

      <div
        class="rounded-2xl p-6 w-56 h-40 flex flex-col items-center justify-center border border-white/30"
      >
        <!-- Icône anniversaire -->
        <img src="/birth.svg" alt="Age" class="w-10 h-10 mb-3 opacity-80" />

        <!-- Texte âge -->
        <p class="text-xl font-bold">{{ analysisData.age }} ans</p>
      </div>
    </div>

    <!-- Section Emotions -->
    <div class="flex flex-col items-center text-white">
      <h3 class="text-xl font-semibold mb-4 text-gray-300">Emotions</h3>

      <div
        class="rounded-2xl p-6 w-56 h-40 flex flex-col items-center justify-center border border-white/30"
      >
        <!-- Icône émotion -->
        <img src="/emotions.svg" alt="Emotions" class="w-10 h-10 mb-3 opacity-80" />

        <!-- Liste des émotions -->
        <div class="text-center space-y-1">
          <p class="text-sm">
            <span class="font-medium"
              >{{ analysisData.emotions[0].emotionName || 'Aucune émotion' }} :</span
            >
            <span class="text-green-400">{{ analysisData.emotions[0].percentage }}%</span>
          </p>
          <p class="text-sm">
            <span class="font-medium">{{ analysisData.emotions[1].emotionName }} :</span>
            <span class="text-yellow-400">{{ analysisData.emotions[1].percentage }}%</span>
          </p>
          <p class="text-sm">
            <span class="font-medium"
              >{{ analysisData.emotions[2].emotionName || 'Aucune émotion' }} :</span
            >
            <span class="text-gray-400">{{ analysisData.emotions[2].percentage }}%</span>
          </p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'

const analysisData = ref({})
onMounted(() => {
  const eventSource = new EventSource('http://localhost:5000/events')

  eventSource.onmessage = (event) => {
    analysisData.value = JSON.parse(event.data)
  }
})
</script>
